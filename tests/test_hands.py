"""브라우저 손 검출(static/hands.js)에서 서버 인식까지 이어서, 지문자 손이 반전되지 않고 글자가 맞게 나오는지 확인한다.

  uv run python tests/test_hands.py   (학습 프로젝트가 옆 폴더에 있어야 함: datasets/output_video의 글자 영상 32개)

hands.js와 같은 HandLandmarker(같은 mediapipe 1.1.1, 같은 모델 파일, 같은 설정)를 Python으로 돌린다.
HandLandmarker는 받은 화면에서 보이는 모양으로 좌우를 붙여서, 반전한 프레임의 오른손을 "Left"라고 한다.
hands.js가 이 라벨을 바꿔 넣지 않으면 서버가 오른손을 반전해 정답률이 18%로 떨어졌다(2026-11-19).
hands.js의 라벨 처리를 바꾸면 아래 BROWSER_LABEL도 같이 바꾼다.
"""
import glob
import sys
from pathlib import Path

import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = sorted(glob.glob(str(ROOT.parent / "hearing_impaired_helper_make_model" / "datasets" / "output_video" / "*" / "*.mp4")))
sys.path.insert(0, str(ROOT / "src"))
from model_api_test import main  # noqa: E402

BROWSER_LABEL = {"Right": "left", "Left": "right"}  # hands.js: HandLandmarker 라벨 → 보내는 키
W = 640
V = mp.tasks.vision


def landmarker():
    return V.HandLandmarker.create_from_options(V.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(main.BASE_DIR / "models" / "hand_landmarker.task")),
        running_mode=V.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.5))


def browser_message(lmk, img, ts_ms):
    """hands.js의 detect와 같다: 폭 640으로 줄이고 좌우 반전 → 검출 → {right, left, w, h}"""
    h = round(W * img.shape[0] / img.shape[1])
    frame = cv2.flip(cv2.resize(img, (W, h), interpolation=cv2.INTER_AREA), 1)
    r = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts_ms)
    msg = {"right": None, "left": None, "w": W, "h": h}
    for lm, hd in zip(r.hand_landmarks, r.handedness):
        if hd[0].category_name in BROWSER_LABEL:
            msg[BROWSER_LABEL[hd[0].category_name]] = [[p.x, p.y] for p in lm]
    return msg


def run_videos(version, flip_video):
    """→ (지문자 손이 기대한 키로 온 비율, 확정 응답 중 정답 비율). flip_video=True면 왼손으로 하는 사람"""
    want = "left" if flip_video else "right"
    if version not in main.interpreters:
        main.interpreters[version] = main.initialize_model(version)
    hands = matched = stable = correct = 0
    for path in VIDEOS:
        label = Path(path).parent.name
        state = main.new_state(version)
        cap, i = cv2.VideoCapture(path), 0
        with landmarker() as lmk:
            while (ok := cap.read())[0]:
                img = cv2.flip(ok[1], 1) if flip_video else ok[1]
                msg = browser_message(lmk, img, i * 1000 // 30)  # 영상은 30fps
                i += 1
                if msg["right"] or msg["left"]:
                    hands += 1
                    matched += msg[want] is not None
                out = main.process_hands(msg, state)
                if out.get("stable"):
                    stable += 1
                    correct += out["action"] == label
        cap.release()
    return matched / hands, correct / stable


def check(version, flip_video):
    matched, acc = run_videos(version, flip_video)
    print(f"  {version} {'왼손' if flip_video else '오른손'}: 손이 맞는 쪽으로 옴 {matched:.1%}, 확정 정답률 {acc:.1%}")
    assert matched > 0.95, matched
    assert acc > 0.85, acc


def test_right_hand_v2_1():
    check("v2.1", flip_video=False)


def test_left_hand_v2_1():
    check("v2.1", flip_video=True)


if __name__ == "__main__":
    if not VIDEOS:
        sys.exit("글자 영상 없음: ../hearing_impaired_helper_make_model/datasets/output_video/*/*.mp4")
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
