"""서버의 지문자 특징이 학습 프로젝트(hearing_impaired_helper_make_model)의 웹캠 계산과 같은지 확인한다.

  uv run python tests/test_features.py   (학습 프로젝트가 옆 폴더에 있어야 함)

서버는 브라우저가 좌우 반전한 프레임에서 찾은 0~1 좌표를 받는다. 학습 쪽 v2는 반전하지 않은 원본의 픽셀 좌표,
v1은 원본의 0~1 좌표를 쓴다. 같은 손이면 두 경로의 특징과 고른 손이 같아야 한다.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT.parent / "hearing_impaired_helper_make_model" / "sing_lang_trans"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(TRAIN))
from model_api_test import main  # noqa: E402
from modules.features import hand_features, letter_features, mirror_xy, select_groups  # noqa: E402

V2_GROUPS = ["base", "lengths"]  # 학습 프로젝트 models/hand_model_training_report.json의 feature_groups


def assert_close(got, want):
    """벡터·길이 비율은 1e-4, 각도는 0.05도까지 (일직선 관절은 arccos가 float 오차를 키운다)"""
    angle = np.zeros(len(want), dtype=bool)
    angle[40:55] = True
    np.testing.assert_allclose(got[~angle], want[~angle], atol=1e-4)
    np.testing.assert_allclose(got[angle], want[angle], atol=0.05)


def train_v2(raw_right, raw_left, w, h):
    """학습 프로젝트 webcam_test_model_1.py의 v2 경로: 높이 든 손, 원본 픽셀 좌표, 왼손은 반전"""
    if raw_right is None or (raw_left is not None and raw_left[0, 1] < raw_right[0, 1]):
        hand, mirror = raw_left, True
    else:
        hand, mirror = raw_right, False
    xy = hand * [w, h]
    return select_groups(letter_features(mirror_xy(xy) if mirror else xy), V2_GROUPS)


def server(raw_right, raw_left, version, w=1.0, h=1.0):
    """브라우저가 보내는 좌표(반전 프레임 기준)로 서버 경로를 탄다"""
    flip = lambda p: None if p is None else np.column_stack([1 - p[:, 0], p[:, 1]]).astype(np.float32)
    joint, mirror = main.pick_hand(flip(raw_right), flip(raw_left), version)
    return main.process_hand_landmarks(joint, mirror, version, w, h)


def test_v2_matches_training():
    rng = np.random.default_rng(0)
    for _ in range(200):
        right, left = (rng.random((21, 2)).astype(np.float32) for _ in range(2))
        for r, l in ((right, None), (None, left), (right, left)):
            got, want = server(r, l, "v2", 640, 360), train_v2(r, l, 640, 360)
            assert got.shape == (75,)
            assert_close(got, want)


def test_v1_matches_training():
    rng = np.random.default_rng(1)
    for _ in range(200):
        right = rng.random((21, 2)).astype(np.float32)
        # v1 학습 경로: 반전 프레임의 오른손 0~1 좌표 그대로 (예전 서버와 같음)
        want = hand_features(np.column_stack([1 - right[:, 0], right[:, 1]]))
        assert_close(server(right, None, "v1"), want)
        # 왼손만 보이면 반전 프레임 좌표를 다시 반전해 쓴다 = 원본 좌표 그대로 (예전 서버와 같음)
        assert_close(server(None, right, "v1"), hand_features(right))


def test_degenerate_hand_is_finite():
    # 관절이 전부 한 점이면 길이 0: NaN 없이 0이 나와야 한다
    joint = np.full((21, 2), 0.5, dtype=np.float32)
    assert np.isfinite(main.process_hand_landmarks(joint, False, "v2", 640, 360)).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
