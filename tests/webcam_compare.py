"""웹캠 하나의 같은 프레임을 두 모델로 동시에 인식해 창 두 개에 띄운다 (왼쪽 창 TFLite 기존 모델, 오른쪽 창 KERNIX 새 모델).

  uv run python tests/webcam_compare.py [카메라 번호 또는 동영상 경로]   (기본 0, q 또는 Esc로 종료)

브라우저(static/hands.js)처럼 좌우 반전 → MediaPipe로 랜드마크를 뽑고, 서버의 process_hands를 그대로 쓴다: 10프레임 → 확률 0.9 이상 같은 글자 3회 연속이면 확정.
창마다 손 검출기를 따로 둬서 서버의 연결 두 개와 같다. KERNIX는 KERNIX_ROOT 체크아웃 (먼저 make python-lib).
처음 실행할 때 macOS가 터미널의 카메라 권한을 묻는다.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_api_test import main  # noqa: E402
from model_api_test.modules import holistic_module as hm  # noqa: E402

TITLES = {"tflite": "TFLite (old model)", "kernix": "KERNIX (2026-11-05 model)"}  # 창 제목은 OpenCV가 한글을 못 씀
FONT = ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", 30)
infer_us = {}  # 런타임별 마지막 추론 시간


def timed_predict(model, x, _predict=main.predict_action):
    t = time.perf_counter()
    y = _predict(model, x)
    infer_us[main.MODEL_RUNTIME] = (time.perf_counter() - t) * 1e6
    return y


main.predict_action = timed_predict  # process_hands가 이 이름으로 부름: 추론 시간만 따로 잼


def load():
    models, states = {}, {}
    for rt in TITLES:
        main.MODEL_RUNTIME = rt
        main.KNX_PATH = main.NEW_KNX_PATH  # 이 창은 서로 다른 모델 비교용: KERNIX 창은 11월 5일 모델
        models[rt] = main.initialize_model()
        states[rt] = {**main.new_state("v1"), "detector": hm.HolisticDetector(min_detection_confidence=0.3)}
    return models, states


def step(frame, models, states):
    """카메라 프레임 하나 → {런타임: 결과를 그린 이미지}"""
    view = cv2.flip(frame, 1)
    out = {}
    for rt in TITLES:
        main.MODEL_RUNTIME, main.interpreters["v1"] = rt, models[rt]
        infer_us.pop(rt, None)
        r = main.process_hands(hands_message(states[rt]["detector"], view), states[rt])
        out[rt] = draw(view.copy(), rt, r)
    return out


def hands_message(detector, img):
    """브라우저가 /ws/sign에 보내는 것과 같은 {"right", "left"} 메시지"""
    detector.findHolistic(img, draw=False)
    res = detector.results
    hand = lambda h: None if h is None else [[lm.x, lm.y] for lm in h.landmark]
    return {"right": hand(res.right_hand_landmarks), "left": hand(res.left_hand_landmarks)}


def draw(img, rt, r):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    if not r["detected"]:
        lines, color = ["손 없음"], (200, 200, 200)
    elif r.get("frames_collected", 0) < main.seq_length:
        lines, color = [f"프레임 모으는 중 {r['frames_collected']}/{main.seq_length}"], (200, 200, 200)
    else:
        lines = [f"확정: {r['action']}", f"확률 {r['confidence']:.3f}" + ("  (안정)" if r["stable"] else ""),
                 f"추론 {infer_us.get(rt, 0):.0f} us"]  # 글꼴에 µ가 없음
        color = (0, 220, 0) if r["stable"] else (255, 200, 0)
    d.rectangle((0, 0, pil.width, 40 + 38 * len(lines)), fill=(0, 0, 0))
    d.text((12, 6), TITLES[rt], font=FONT, fill=(255, 255, 255))
    for i, line in enumerate(lines):
        d.text((12, 44 + 38 * i), line, font=FONT, fill=color)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def run(source):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"카메라/동영상을 열 수 없음: {source} (macOS 설정 > 개인정보 보호 > 카메라에서 터미널 권한 확인)")
    models, states = load()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for i, (rt, img) in enumerate(step(frame, models, states).items()):
                cv2.imshow(TITLES[rt], img)
                cv2.moveWindow(TITLES[rt], i * (img.shape[1] + 10), 0)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for s in states.values():
            s["detector"].close()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "0")
