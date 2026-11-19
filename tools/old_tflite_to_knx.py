"""기존 모델(.tflite, 원본 .h5 없음)을 KERNIX .knx로 변환.

  uv run python tools/old_tflite_to_knx.py [출력.knx]   (기본 src/model_api_test/models/multi_hand_gesture_classifier_old.knx)

.tflite에서 가중치를 꺼내 같은 구성의 Keras 모델(LSTM 64 relu → Dense 32 relu → Dense 31 softmax, 입력 10×55)에 넣고
KERNIX 변환기로 내보낸 뒤, 같은 입력에서 .tflite와 출력이 같은지 확인한다. KERNIX는 KERNIX_ROOT 체크아웃의 tools/를 쓴다.
"""
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.lite.tools import flatbuffer_utils

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "src" / "model_api_test" / "models"
TFLITE = MODELS / "multi_hand_gesture_classifier.tflite"
KERNIX_ROOT = Path(os.environ.get("KERNIX_ROOT", ROOT.parents[1] / "KERNIX"))


def tflite_weights(path):
    """모든 하위 그래프(LSTM은 WHILE 안쪽)의 상수 텐서를 모양으로 찾음 → Keras 순서(입력×출력)로 전치"""
    m = flatbuffer_utils.read_model(str(path))
    by_shape = {}
    for sg in m.subgraphs:
        for t in sg.tensors:
            data = m.buffers[t.buffer].data
            if data is not None and len(data) > 64:
                by_shape.setdefault(tuple(int(d) for d in t.shape), np.frombuffer(bytes(data), np.float32).reshape(t.shape))
    w = lambda *s: by_shape[s]  # noqa: E731  모양이 겹치면 KeyError 대신 잘못 고를 수 있어 아래에서 출력으로 확인
    return [w(256, 55).T, w(256, 64).T, w(256),  # LSTM: kernel, recurrent_kernel, bias (게이트 i f c o)
            w(32, 64).T, w(32), w(31, 32).T, w(31)]


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else MODELS / "multi_hand_gesture_classifier_old.knx"
    model = tf.keras.Sequential([tf.keras.Input((10, 55)), tf.keras.layers.LSTM(64, activation="relu"),
                                 tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dense(31, activation="softmax")])
    model.set_weights(tflite_weights(TFLITE))

    interp = tf.lite.Interpreter(model_path=str(TFLITE))
    interp.allocate_tensors()
    i, o = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]
    x = np.random.default_rng(0).standard_normal((64, 10, 55)).astype(np.float32)
    want = []
    for s in x:
        interp.set_tensor(i, s[None])
        interp.invoke()
        want.append(interp.get_tensor(o)[0].copy())
    err = np.abs(np.asarray(model(x)) - np.stack(want)).max()
    if err > 1e-4:  # 가중치를 잘못 골랐거나 활성화가 다름: 틀린 모델을 내보내지 않음
        sys.exit(f"Keras 재구성이 .tflite와 다름 (최대 오차 {err:.2e})")

    sys.path.insert(0, str(KERNIX_ROOT / "tools"))
    from export_keras import export
    export(model, str(out))
    print(f"{TFLITE.name} → {out} (Keras 재구성 오차 {err:.1e}). 확인: python $KERNIX_ROOT/tools/verify.py {TFLITE} {out}")


if __name__ == "__main__":
    main()
