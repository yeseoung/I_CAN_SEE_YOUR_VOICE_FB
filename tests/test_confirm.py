"""글자 확정 규칙 확인: 확률 0.7 이상 예측이 repeat번 연속이면, 그중 확률이 가장 높은 글자로 확정한다.

  uv run python tests/test_confirm.py   (모델 없이 돈다)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_api_test import main  # noqa: E402

HAND = np.random.default_rng(0).random((21, 2)).tolist()


def run(version, preds):
    """preds: 프레임마다 (글자, 확률). 10프레임을 먼저 채운 뒤 하나씩 넣고 마지막 응답을 돌려준다"""
    queue = list(preds)

    def fake_predict(_model, _x):
        letter, p = queue.pop(0)
        y = np.full(len(main.actions), (1 - p) / (len(main.actions) - 1), dtype=np.float32)
        y[main.actions.index(letter)] = p
        return y

    main.predict_action = fake_predict
    main.interpreters[version] = None
    state = {"version": version, "seq": [], "action_seq": [], "current_action": "?", "current_conf": 0.0}
    msg = {"right": HAND, "left": None, "w": 640, "h": 480}
    for _ in range(main.seq_length - 1):
        main.process_hands(msg, state)
    return [main.process_hands(msg, state) for _ in preds]


def test_different_letters_pick_highest():
    # v2(3번 연속): ㄱ 0.75, ㅊ 0.9, ㄱ 0.8 → 확률이 가장 높은 ㅊ
    out = run("v2", [("ㄱ", 0.75), ("ㅊ", 0.9), ("ㄱ", 0.8)])
    assert [r["stable"] for r in out] == [False, False, True]
    assert out[-1]["action"] == "ㅊ"


def test_window_slides():
    # v1(2번 연속): 최근 두 예측만 본다. ㅊ 0.95가 창에서 빠지면 ㄱ
    out = run("v1", [("ㅊ", 0.95), ("ㄱ", 0.8), ("ㄱ", 0.85)])
    assert [r["action"] for r in out] == ["?", "ㅊ", "ㄱ"]


def test_below_threshold_resets():
    # 0.7 미만이 끼면 연속이 끊긴다
    out = run("v1", [("ㄱ", 0.9), ("ㄱ", 0.6), ("ㄱ", 0.9)])
    assert [r["stable"] for r in out] == [False, False, False]
    assert out[1]["action"] == "?"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
