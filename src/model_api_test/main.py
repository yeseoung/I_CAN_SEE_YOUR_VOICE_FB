import asyncio
import threading
from contextlib import asynccontextmanager

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from model_api_test.modules import holistic_module as hm
from model_api_test.modules.utils import Vector_Normalization
from PIL import ImageFont, ImageDraw, Image

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "multi_hand_gesture_classifier.tflite"
STATIC_DIR = BASE_DIR / "static"

# 기존 코드에 정의되어 있어야 하는 항목
actions = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ',
    'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ',
    'ㅡ', 'ㅣ',
    'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ'
    ]

seq_length = 10

def initialize_detector_and_model():
    """MediaPipe 홀리스틱 모델과 TFLite 모델 초기화"""

    detector = hm.HolisticDetector(
        min_detection_confidence=0.3
    )

    interpreter = tf.lite.Interpreter(
        model_path=str(MODEL_PATH)
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("========== TFLite Model ==========")
    print("Input shape :", input_details[0]['shape'])
    print("Input dtype :", input_details[0]['dtype'])
    print("Output shape:", output_details[0]['shape'])
    print("Output dtype:", output_details[0]['dtype'])
    print("==================================")

    return detector, interpreter

def process_hand_landmarks(right_hand_lmList):
    """손 랜드마크 처리 및 벡터 정규화"""

    joint = np.zeros((21, 2), dtype=np.float32)

    for j, lm in enumerate(right_hand_lmList.landmark):
        if j >= 21:
            break

        joint[j] = [lm.x, lm.y]

    vector, angle_label = Vector_Normalization(joint)

    result = np.concatenate([
        vector.flatten(),
        angle_label.flatten()
    ])

    return result.astype(np.float32)

def predict_action(interpreter, input_data):
    """TFLite 모델을 사용하여 동작 예측"""

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(
        input_details[0]['index'],
        input_data
    )

    interpreter.invoke()

    y_pred = interpreter.get_tensor(
        output_details[0]['index']
    )

    return y_pred[0]


detector = None
interpreter = None

# 공유 MediaPipe 검출기와 TFLite 인터프리터의 동시 접근 방지
inference_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, interpreter

    detector, interpreter = initialize_detector_and_model()

    yield

    detector = None
    interpreter = None


app = FastAPI(lifespan=lifespan)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


def process_frame(frame_bytes: bytes, state: dict) -> dict:
    """
    JPEG 프레임 하나를 받아 수어를 추론한다.
    state는 WebSocket 연결별로 독립적으로 유지한다.
    """

    # JPEG -> OpenCV 이미지
    encoded = np.frombuffer(frame_bytes, dtype=np.uint8)
    img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("유효하지 않은 JPEG 프레임입니다.")

    # 기존 코드와 동일하게 좌우 반전
    img = cv2.flip(img, 1)

    # 공유 검출기와 모델은 동시에 사용하지 않도록 보호
    with inference_lock:

        # 랜드마크 검출
        img = detector.findHolistic(img, draw=False)

        _, right_hand_lmList = detector.findRighthandLandmark(img)

        if right_hand_lmList is None:
            # 손이 사라지면 연속 프레임 기록 초기화
            state["seq"].clear()
            state["action_seq"].clear()
            state["current_action"] = "?"
            state["current_conf"] = 0.0

            return {
                "detected": False,
                "action": "?",
                "confidence": 0.0,
                "stable": False,
            }

        # 랜드마크 -> 기존 모델 입력 특징
        d = process_hand_landmarks(right_hand_lmList)

        state["seq"].append(d)

        if len(state["seq"]) > seq_length:
            state["seq"] = state["seq"][-seq_length:]

        # 충분한 프레임이 모이지 않았으면 대기
        if len(state["seq"]) < seq_length:
            return {
                "detected": True,
                "action": state["current_action"],
                "confidence": state["current_conf"],
                "stable": False,
                "frames_collected": len(state["seq"]),
            }

        # 모델 입력 shape: [1, 10, 특징 수]
        input_data = np.expand_dims(
            np.asarray(state["seq"], dtype=np.float32),
            axis=0,
        )

        y_pred = predict_action(interpreter, input_data)

        i_pred = int(np.argmax(y_pred))
        conf = float(y_pred[i_pred])

        state["current_conf"] = conf

        if conf >= 0.9:
            action = actions[i_pred]
            state["action_seq"].append(action)

            if len(state["action_seq"]) > 5:
                state["action_seq"] = state["action_seq"][-5:]

            recent = state["action_seq"]

            # 동일한 동작이 3회 연속 나왔는지 확인
            stable = (
                len(recent) >= 3
                and recent[-1] == recent[-2] == recent[-3]
            )

            if stable:
                state["current_action"] = action

        else:
            # 낮은 신뢰도의 예측은 연속 일치 판정에서 제외
            state["action_seq"].clear()
            state["current_action"] = "?"
            stable = False

        return {
            "detected": True,
            "action": state["current_action"],
            "confidence": conf,
            "stable": stable,
            "frames_collected": len(state["seq"]),
        }


@app.websocket("/ws/sign")
async def sign_recognition(websocket: WebSocket):
    await websocket.accept()

    # 연결마다 별도 상태 생성
    state = {
        "seq": [],
        "action_seq": [],
        "current_action": "?",
        "current_conf": 0.0,
    }

    frame_id = 0

    try:
        while True:
            # 클라이언트에서 전송한 JPEG 바이너리 수신
            frame_bytes = await websocket.receive_bytes()

            # 이미지 처리 및 추론은 이벤트 루프 밖에서 실행
            result = await asyncio.to_thread(
                process_frame,
                frame_bytes,
                state,
            )

            frame_id += 1

            result["frame_id"] = frame_id

            # 클라이언트로 결과 반환
            await websocket.send_json(result)

    except WebSocketDisconnect:
        print("클라이언트 연결 종료")

    except ValueError as e:
        await websocket.close(
            code=1003,
            reason=str(e)[:120],
        )

    except Exception as e:
        print("프레임 처리 오류:", e)

        try:
            await websocket.close(code=1011)
        except RuntimeError:
            pass