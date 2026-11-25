"""글자 확정 규칙 확인: 로컬 웹캠(학습 프로젝트 webcam_test_model_1.py)과 같아야 한다.
확률 기준(v1 0.9, v2·v2.1 0.8) 이상 예측만 기록하고, 최근 3번이 같은 글자면 확정, 직전 확정과 다를 때만 새 글자.

  uv run python tests/test_confirm.py   (모델 없이 돈다. 마지막 테스트는 학습 프로젝트가 옆 폴더에 있어야 함)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from model_api_test import main  # noqa: E402

HAND = np.random.default_rng(0).random((21, 2)).tolist()


def run(version, preds):
    """preds: 프레임마다 (글자, 확률). 10프레임을 먼저 채운 뒤 하나씩 넣고 응답 목록을 돌려준다"""
    queue = list(preds)

    def fake_predict(_model, _x):
        letter, p = queue.pop(0)
        y = np.full(len(main.actions), (1 - p) / (len(main.actions) - 1), dtype=np.float32)
        y[main.actions.index(letter)] = p
        return y

    main.predict_action = fake_predict
    main.interpreters[version] = None
    state = main.new_state(version)
    msg = {"right": HAND, "left": None, "w": 640, "h": 480}
    for _ in range(main.seq_length - 1):
        main.process_hands(msg, state)
    return [main.process_hands(msg, state) for _ in preds]


def test_three_same_letters_confirm_once():
    out = run("v2.1", [("ㄱ", 0.9)] * 5)
    assert [r["stable"] for r in out] == [False, False, True, True, True]
    assert [r["new"] for r in out] == [None, None, "ㄱ", None, None]  # 들고 있어도 한 번만


def test_different_letters_do_not_confirm():
    out = run("v2.1", [("ㄱ", 0.9), ("ㅊ", 0.95), ("ㄱ", 0.9)])
    assert not any(r["stable"] for r in out)


def test_low_confidence_is_skipped_not_reset():
    # 0.8 미만은 기록하지 않을 뿐 연속을 끊지 않는다: ㄱ, (건너뜀), ㄱ, ㄱ → 네 번째에서 확정
    out = run("v2.1", [("ㄱ", 0.9), ("ㄱ", 0.75), ("ㄱ", 0.9), ("ㄱ", 0.9)])
    assert [r["new"] for r in out] == [None, None, None, "ㄱ"]
    assert out[1]["action"] == "?"


def test_v1_threshold_is_0_9():
    assert not any(r["stable"] for r in run("v1", [("ㄱ", 0.85)] * 4))


def test_matches_local_webcam():
    # 무작위 예측열을 서버와 학습 프로젝트의 LetterStabilizer에 똑같이 넣어, 새로 확정되는 글자가 같은지
    sys.path.insert(0, str(ROOT.parent / "hearing_impaired_helper_make_model" / "sing_lang_trans"))
    from modules.jamo import LetterStabilizer

    rng = np.random.default_rng(0)
    for _ in range(200):
        preds = [(str(rng.choice(["ㄱ", "ㄴ", "ㅊ"])), float(rng.choice([0.5, 0.79, 0.81, 0.95]))) for _ in range(30)]
        stab, want = LetterStabilizer(), []
        for letter, p in preds:
            want.append(stab.push(letter)[1] if p >= 0.8 else None)  # webcam_test_model_1.py: conf >= min_conf일 때만 push
        assert [r["new"] for r in run("v2.1", preds)] == want


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
