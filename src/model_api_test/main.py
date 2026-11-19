import asyncio
import os
import secrets
import threading
from contextlib import asynccontextmanager

import numpy as np
from model_api_test.modules.utils import Bone_Length_Ratio, Vector_Normalization

from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "multi_hand_gesture_classifier.tflite"
# MODEL_RUNTIME=kernix: 같은 기존 모델을 KERNIX로 실행 (tools/old_tflite_to_knx.py로 변환, 결과는 TFLite와 같음).
# 11월 5일 재학습 모델은 KNX_PATH=.../multi_hand_gesture_classifier.knx
MODEL_RUNTIME = os.environ.get("MODEL_RUNTIME", "tflite")
KNX_PATH = Path(os.environ.get("KNX_PATH", BASE_DIR / "models" / "multi_hand_gesture_classifier_old.knx"))
NEW_KNX_PATH = BASE_DIR / "models" / "multi_hand_gesture_classifier.knx"
KERNIX_ROOT = Path(os.environ.get("KERNIX_ROOT", BASE_DIR.parents[3] / "KERNIX"))
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

# 사용자가 고르는 지문자 모델별 확정 기준: 확률 conf 이상 예측이 repeat번 연속 나오면,
# 그중 확률이 가장 높은 글자로 확정한다 (연속 예측끼리 글자가 달라도 된다).
# v2는 hearing_impaired_helper_make_model의 2026-11-17 배포 모델
MODELS = {
    "v1": {"conf": 0.7, "repeat": 2},
    "v2": {"conf": 0.7, "repeat": 3},
}


def model_path(version):
    """버전과 런타임에 맞는 모델 파일. v1은 MODEL_PATH / KNX_PATH 설정을 따른다."""

    kernix = MODEL_RUNTIME == "kernix"
    if version == "v2":
        return BASE_DIR / "models" / ("multi_hand_gesture_classifier_v2." + ("knx" if kernix else "tflite"))
    return KNX_PATH if kernix else MODEL_PATH


def initialize_model(version="v1"):
    """TFLite 또는 KERNIX 분류 모델 초기화 (모든 연결이 공유)"""

    print(f"[{version}] {model_path(version).name}")

    if MODEL_RUNTIME == "kernix":
        import sys
        sys.path.insert(0, str(KERNIX_ROOT / "python"))
        from kernix import Model  # 없으면: cd $KERNIX_ROOT && make python-lib

        model = Model(str(model_path(version)))
        print("========== KERNIX Model ==========")
        print("Input shape :", model.in_shape, " Output shape:", model.out_shape)
        print("==================================")
        return model

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path(version))
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

    return interpreter

def parse_hand(points):
    """클라이언트가 보낸 손 하나(21개 [x, y]) → (21, 2) 배열. 없으면 None, 형식이 틀리면 ValueError"""

    if points is None:
        return None

    try:
        joint = np.asarray(points, dtype=np.float32)
    except (TypeError, ValueError):
        raise ValueError("손 랜드마크는 21개 [x, y] 배열이어야 합니다.")

    if joint.shape != (21, 2) or not np.isfinite(joint).all():
        raise ValueError("손 랜드마크는 21개 [x, y] 배열이어야 합니다.")

    return joint


def parse_size(msg):
    """v2용 프레임 크기 (w, h). 양수가 아니면 ValueError"""

    try:
        w, h = float(msg["w"]), float(msg["h"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("v2 모델은 프레임 크기 w, h가 필요합니다.")

    if not (np.isfinite(w) and np.isfinite(h) and w > 0 and h > 0):
        raise ValueError("v2 모델은 프레임 크기 w, h가 필요합니다.")

    return w, h


def pick_hand(right, left, version):
    """→ (지문자 손, 왼손이라 반전할지). v1: 오른손 우선. v2: 두 손이 보이면 손목이 더 높은 손(v2 학습 규칙)"""

    if right is None or (version == "v2" and left is not None and left[0, 1] < right[0, 1]):
        return left, left is not None

    return right, False


def process_hand_landmarks(joint, mirror=False, version="v1", w=1.0, h=1.0):
    """손 랜드마크 (21, 2) → 모델 입력 특징. v1: 55차원, v2: 55 + 뼈 길이 비율 20 = 75차원"""

    joint = joint.copy()

    if version == "v2":
        # v2는 반전하지 않은 원본 프레임의 픽셀 좌표로 학습했다. 받은 좌표는 반전 프레임 기준이라 되돌린다
        joint[:, 0] = 1.0 - joint[:, 0]
        joint *= np.array([w, h], dtype=np.float32)

    if mirror:
        # 반대 손은 좌우 반전해 모델이 학습한 손 모양으로 맞춘다 (특징이 관절 간 차이라 x 부호만 바꾸면 된다)
        joint[:, 0] = -joint[:, 0]

    vector, angle_label = Vector_Normalization(joint)

    features = [vector.flatten(), angle_label.flatten()]

    if version == "v2":
        features.append(Bone_Length_Ratio(joint))

    return np.concatenate(features).astype(np.float32)

def predict_action(interpreter, input_data):
    """TFLite 또는 KERNIX 모델로 동작 예측"""

    if MODEL_RUNTIME == "kernix":
        return interpreter(input_data)[0]

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


interpreters = {}  # 버전 -> 모델

# 공유 분류 모델의 동시 접근 방지 (검출기는 연결마다 따로라 잠그지 않는다)
inference_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    for version in MODELS:
        interpreters[version] = initialize_model(version)

    yield

    interpreters.clear()


app = FastAPI(lifespan=lifespan)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "app.html")


@app.get("/models/hand_landmarker.task")
async def hand_model():
    # 브라우저 손 검출기가 서버에 있던 것과 같은 모델을 쓰게 한다
    return FileResponse(BASE_DIR / "models" / "hand_landmarker.task")


@app.get("/recognition")
async def recognition():
    # 기존 단일 사용자 인식 데모는 그대로 남겨 둔다
    return FileResponse(STATIC_DIR / "index.html")


# --- 더미 로그인 / 회의: 모두 메모리에만 저장(서버 재시작하면 사라짐) ---
DUMMY_USERS = {
    "alice": "1234", "bob": "1234", "demo": "demo",
    # 1:1 통화 테스트용 관리자 계정 (권한 차이는 아직 없음)
    "admin1": "admin1234", "admin2": "admin1234",
}
ADMINS = {"admin1", "admin2"}
# 세션 토큰 -> 아이디. 로그인할 때마다 새 토큰이라 탭마다 다른 계정으로 쓸 수 있다
sessions: dict[str, str] = {}
meetings: dict[str, str] = {}  # 회의 코드 -> 개설자 이름
# 회의 코드 -> {peerId: {"ws": WebSocket, "name": str}}
rooms: dict[str, dict[str, dict]] = {}


def session_user(req: Request) -> str | None:
    return sessions.get(req.headers.get("authorization", "").removeprefix("Bearer "))


def unauthorized():
    return JSONResponse({"error": "로그인이 필요합니다."}, status_code=401)


@app.post("/api/login")
async def login(req: Request):
    body = await req.json()
    name = str(body.get("username", "")).strip()
    password = str(body.get("password", ""))

    if DUMMY_USERS.get(name) == password:
        token = secrets.token_urlsafe(32)
        sessions[token] = name
        role = "admin" if name in ADMINS else "user"
        return {"ok": True, "name": name, "role": role, "token": token}

    return JSONResponse(
        {"ok": False, "error": "아이디 또는 비밀번호가 올바르지 않습니다."},
        status_code=401,
    )


@app.post("/api/logout")
async def logout(req: Request):
    sessions.pop(req.headers.get("authorization", "").removeprefix("Bearer "), None)
    return {"ok": True}


@app.post("/api/meetings")
async def create_meeting(req: Request):
    host = session_user(req)
    if host is None:
        return unauthorized()
    code = secrets.token_hex(3)  # 6자리 회의 코드
    meetings[code] = host
    return {"code": code, "host": host}


@app.get("/api/meetings")
async def list_meetings(req: Request):
    if session_user(req) is None:
        return unauthorized()
    # 개설된 회의 목록. rooms에 실시간 참가자가 있으면 '진행 중'으로 표시한다.
    items = [
        {
            "code": code,
            "host": host,
            "participants": len(rooms.get(code, {})),
            "live": len(rooms.get(code, {})) > 0,
        }
        for code, host in meetings.items()
    ]
    items.sort(key=lambda m: (not m["live"], -m["participants"]))
    return {"meetings": items}


@app.get("/api/meetings/{code}")
async def get_meeting(code: str, req: Request):
    if session_user(req) is None:
        return unauthorized()
    if code in meetings:
        return {"exists": True, "host": meetings[code]}
    return JSONResponse({"exists": False}, status_code=404)


async def room_broadcast(code: str, message: dict, exclude: str | None = None):
    for pid, peer in list(rooms.get(code, {}).items()):
        if pid == exclude:
            continue
        try:
            await peer["ws"].send_json(message)
        except Exception:
            pass


@app.websocket("/ws/room/{code}")
async def room_ws(websocket: WebSocket, code: str):
    """WebRTC 시그널링 중계 + 참가/이탈 알림 + 수어 자막 중계."""
    # shortcut: 토큰이 URL 쿼리라 서버 접속 로그에 남는다. 실제 배포 전에 첫 메시지 인증으로 바꾼다
    name = sessions.get(websocket.query_params.get("token", ""))
    if name is None or code not in meetings:
        await websocket.close(code=1008)
        return

    await websocket.accept()

    peer_id = secrets.token_hex(4)
    room = rooms.setdefault(code, {})

    # 새로 들어온 사람에게 기존 참가자 목록을 알려 주고,
    # 기존 참가자들에게는 새 참가자를 알린다.
    existing = [{"peerId": pid, "name": p["name"]} for pid, p in room.items()]
    room[peer_id] = {"ws": websocket, "name": name}

    await websocket.send_json(
        {"type": "welcome", "peerId": peer_id, "peers": existing}
    )
    await room_broadcast(
        code,
        {"type": "peer-joined", "peerId": peer_id, "name": name},
        exclude=peer_id,
    )

    try:
        while True:
            msg = await websocket.receive_json()
            kind = msg.get("type")

            if kind == "signal":
                target = room.get(msg.get("to"))
                if target:
                    await target["ws"].send_json(
                        {"type": "signal", "from": peer_id, "data": msg.get("data")}
                    )

            elif kind == "caption":
                await room_broadcast(
                    code,
                    {
                        "type": "caption",
                        "from": peer_id,
                        "name": name,
                        "text": str(msg.get("text", "")),
                    },
                    exclude=peer_id,
                )

    except WebSocketDisconnect:
        pass

    finally:
        room.pop(peer_id, None)
        if not room:
            rooms.pop(code, None)
        # 화상통화 이탈을 다른 참가자에게 알린다
        await room_broadcast(code, {"type": "peer-left", "peerId": peer_id})


def process_hands(msg: dict, state: dict) -> dict:
    """
    클라이언트가 검출한 손 랜드마크 한 프레임 {"right": [[x, y]*21] | null, "left": ..., "w": 640, "h": 360}로
    수어를 추론한다. 좌표는 좌우 반전한 영상 기준(0~1), w·h는 그 영상의 크기(v2만 씀).
    state는 WebSocket 연결별로 독립적으로 유지한다.
    """

    if not isinstance(msg, dict):
        raise ValueError("메시지는 {\"right\", \"left\"} 객체여야 합니다.")

    version = state["version"]
    w, h = parse_size(msg) if version == "v2" else (1.0, 1.0)

    right_hand_lmList, mirror = pick_hand(
        parse_hand(msg.get("right")),
        parse_hand(msg.get("left")),
        version,
    )

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

    # 랜드마크 -> 모델 입력 특징
    d = process_hand_landmarks(right_hand_lmList, mirror, version, w, h)

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

    with inference_lock:
        y_pred = predict_action(interpreters[version], input_data)

    i_pred = int(np.argmax(y_pred))
    conf = float(y_pred[i_pred])

    state["current_conf"] = conf

    settings = MODELS[version]

    if conf >= settings["conf"]:
        # (확률, 글자)로 쌓아 최근 repeat개만 둔다
        state["action_seq"] = (state["action_seq"] + [(conf, actions[i_pred])])[-settings["repeat"]:]

        # 기준 이상 예측이 repeat회 연속이면, 그중 확률이 가장 높은 글자로 확정
        stable = len(state["action_seq"]) == settings["repeat"]

        if stable:
            state["current_action"] = max(state["action_seq"])[1]

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
    # 사용할 지문자 모델은 접속 주소로 고른다: /ws/sign?model=v2 (기본 v1)
    version = websocket.query_params.get("model", "v1")
    if version not in MODELS:
        await websocket.close(code=1008)
        return

    await websocket.accept()

    # 연결마다 별도 상태 생성
    state = {
        "version": version,
        "seq": [],
        "action_seq": [],
        "current_action": "?",
        "current_conf": 0.0,
    }

    frame_id = 0

    try:
        while True:
            # 클라이언트가 검출한 손 랜드마크 JSON 수신 (잘못된 JSON은 ValueError)
            msg = await websocket.receive_json()

            # 추론은 이벤트 루프 밖에서 실행
            result = await asyncio.to_thread(
                process_hands,
                msg,
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