"""기존 TFLite 경로와 KERNIX 경로를 서버가 실제로 부르는 함수(initialize_model, predict_action)로 비교.

  uv run python tests/compare_runtimes.py [DATASET_DIR]

DATASET_DIR: 학습 프로젝트의 datasets/ (기본 ../hearing_impaired_helper_make_model/datasets).
검증 세션(seq_*_1669720403.npy, 마지막 열이 정답 번호)의 모든 표본을 표본 하나씩 넣는다 (서버와 같은 [1, 10, 55]).
KERNIX는 KERNIX_ROOT의 체크아웃을 쓴다 (먼저 cd $KERNIX_ROOT && make python-lib).
KERNIX 쪽 모델은 서버와 같이 KNX_PATH (기본: 같은 기존 모델이라 예측이 100% 일치해야 함, KNX_PATH로 새 모델을 주면 모델 비교).
"""
import glob
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_api_test import main  # noqa: E402

CONF = 0.9  # 서버의 확정 임계값 (main.py의 conf >= 0.9)
VAL_SESSION = "1669720403"


def load(dataset_dir):
    files = sorted(glob.glob(str(Path(dataset_dir) / f"seq_*_{VAL_SESSION}.npy")))
    if not files:
        sys.exit(f"검증 데이터 없음: {dataset_dir}/seq_*_{VAL_SESSION}.npy")
    data = np.concatenate([np.load(f) for f in files])
    x, y = data[..., :-1].astype(np.float32), data[:, 0, -1].astype(int)
    assert x.shape[1:] == (main.seq_length, 55), x.shape
    return x, y


def run(runtime, x):
    """runtime 경로로 표본마다 예측 → (확률 (N, 31), 표본당 시간 중앙값 µs)"""
    main.MODEL_RUNTIME = runtime  # predict_action이 호출 때마다 읽는 전역값
    model = main.initialize_model()
    for s in x[:200]:  # 예열: 쉬지 않고 불러 클럭을 올림
        main.predict_action(model, s[None])
    probs, times = [], []
    for s in x:
        t = time.perf_counter()
        probs.append(main.predict_action(model, s[None]))
        times.append((time.perf_counter() - t) * 1e6)
    return np.asarray(probs), statistics.median(times)


def report(name, p, y, us):
    pred, conf = p.argmax(1), p.max(1)
    sure = conf >= CONF
    print(f"{name:<8} 정확도 {np.mean(pred == y) * 100:6.2f}%   "
          f"확정 {sure.mean() * 100:5.1f}% 중 정확 {np.mean(pred[sure] == y[sure]) * 100:6.2f}%   "
          f"추론 {us:6.1f} µs/표본")


def main_():
    dataset = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "hearing_impaired_helper_make_model" / "datasets"
    x, y = load(dataset)
    print(f"검증 표본 {len(x)}개 (세션 {VAL_SESSION}), 확정 임계값 {CONF}\n")
    tf_p, tf_us = run("tflite", x)
    kx_p, kx_us = run("kernix", x)
    print()
    report("TFLite", tf_p, y, tf_us)
    report("KERNIX", kx_p, y, kx_us)
    agree = np.mean(tf_p.argmax(1) == kx_p.argmax(1)) * 100
    print(f"\n두 경로의 예측 일치 {agree:.2f}% (KERNIX 모델: {main.KNX_PATH.name}), 추론 속도 {tf_us / kx_us:.1f}배")


if __name__ == "__main__":
    main_()
