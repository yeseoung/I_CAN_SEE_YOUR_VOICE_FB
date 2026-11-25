# SORI (소리): Sign Language Video Meetings

A FastAPI server that recognizes Korean Sign Language **fingerspelling** (31 consonants and vowels) from a webcam in real time and shows the result as live captions in a browser video meeting.

- **Recognition:** the browser finds hand landmarks with MediaPipe and sends them (not images) over a WebSocket. The server turns them into joint vectors and angles and classifies the last 10 frames with a fingerspelling model: v1 (LSTM) or v2 (Conv1D), chosen per user during a call (TFLite, or [KERNIX](https://github.com/yeseoung/KERNIX) on Apple Silicon).
- **Meetings:** create or join a meeting by a 6-character code. Video and audio are peer-to-peer WebRTC. The server only relays signaling messages and captions.
- **Either hand works:** the left hand is mirrored to look like a right hand. v1 prefers the right hand; v2 uses whichever hand is raised higher.

> **Status: graduation project prototype.** Logins are hard-coded demo accounts and all meeting data lives in memory. Do not expose it to the internet as is.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- A webcam and a recent Chrome, Edge, Safari or Firefox

## Quick start

```bash
git clone https://github.com/yeseoung/I_CAN_SEE_YOUR_VOICE_FB.git
cd I_CAN_SEE_YOUR_VOICE_FB
uv sync
uv run uvicorn model_api_test.main:app --app-dir src --reload
```

Wait for `Application startup complete.` (loading the models takes a few seconds), then open:

| URL | Page |
|---|---|
| http://127.0.0.1:8000/ | Login, meeting list, video meeting with sign captions |
| http://127.0.0.1:8000/recognition | Single-user recognition demo |

Demo accounts:

| Username | Password | Role |
|---|---|---|
| `admin1` | `admin1234` | admin |
| `admin2` | `admin1234` | admin |
| `alice` | `1234` | user |
| `bob` | `1234` | user |
| `demo` | `demo` | user |

The admin role is only a label on the My Page screen for now. It grants no extra permissions.

### Testing a 1:1 call on one computer

Each browser tab keeps its own login session, so two tabs can be signed in as different users.

1. In tab 1, sign in as `admin1` and click **새 회의 시작** (start new meeting). Note the 6-character code at the top.
2. In tab 2, sign in as `admin2`, enter the code and click **참여** (join).
3. Allow camera and microphone in both tabs. Each tab shows both videos. Fingerspelling captions appear under the signer's video in both tabs. Put the two tabs in separate windows side by side: Chrome slows down timers in a hidden tab, so recognition in a background tab almost stops.

Refreshing a tab signs it out.

Stop the server with `Ctrl+C`.

### Using other devices on your network

Browsers only allow camera access on `https://` or `http://localhost` / `http://127.0.0.1`, so any other address (your LAN IP, `0.0.0.0`) needs HTTPS. For testing, a self-signed certificate is enough (`certs/` is git-ignored):

```bash
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes -keyout certs/key.pem -out certs/cert.pem -days 365 \
  -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:$(ipconfig getifaddr en0)"
uv run uvicorn model_api_test.main:app --app-dir src --host 0.0.0.0 --port 8443 \
  --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
```

Open `https://<your-ip>:8443` and accept the certificate warning once per browser (Chrome: **Advanced → Proceed**). `ipconfig getifaddr en0` is for macOS Wi-Fi; regenerate the certificate if your IP changes. Meetings use a public STUN server only (no TURN), so peers behind strict NATs may fail to connect.

## Optional: KERNIX runtime (Apple Silicon)

[KERNIX](https://github.com/yeseoung/KERNIX) is a C inference runtime for Apple Silicon. It runs the classifier about 5× faster than TFLite per frame.

```bash
# Once: build the KERNIX Python library
git clone https://github.com/yeseoung/KERNIX.git ../KERNIX
(cd ../KERNIX && make python-lib)

# Run this server with KERNIX
MODEL_RUNTIME=kernix KERNIX_ROOT=../KERNIX uv run uvicorn model_api_test.main:app --app-dir src --reload
```

The startup log prints `KERNIX Model` instead of `TFLite Model`.

KERNIX mode runs the same model as TFLite mode: `multi_hand_gesture_classifier_old.knx` is converted from the `.tflite` by `tools/old_tflite_to_knx.py` (its original `.h5` is lost, so the weights are read from the `.tflite`). To run the model retrained on 2026-11-05 instead, set `KNX_PATH=src/model_api_test/models/multi_hand_gesture_classifier.knx`. On the same 6,219 validation samples:

| | TFLite (default) | KERNIX | KERNIX, 2026-11-05 model |
|---|---:|---:|---:|
| Accuracy | 90.45% | 90.45% | 85.46% |
| Accuracy of predictions with confidence ≥ 0.9 | 98.11% | 98.11% | 90.51% |
| Inference per frame (M1) | 70 µs | 14 µs | 14 µs |

The validation split comes from the newer model's training run, so the older model may have seen some of these samples. In practice MediaPipe hand detection dominates frame time, so the speed-up is hard to notice.

To reproduce the table (measured with KERNIX v0.2.2), run both paths through the server's own `initialize_model` and `predict_action` (add `KNX_PATH=...` for the last column):

```bash
uv run python tests/compare_runtimes.py [path/to/hearing_impaired_helper_make_model/datasets]
```

To try the old model (TFLite) and the 2026-11-05 model (KERNIX) live, open two windows on the same webcam frames (`q` or Esc quits):

```bash
uv run python tests/webcam_compare.py [camera index or video file]
```

## Configuration

| Environment variable | Default | Meaning |
|---|---|---|
| `MODEL_RUNTIME` | `tflite` | `tflite` or `kernix` |
| `KERNIX_ROOT` | `../../KERNIX` (relative to the repo root) | Path to a KERNIX checkout (KERNIX mode only) |
| `KNX_PATH` | `src/model_api_test/models/multi_hand_gesture_classifier_old.knx` | Model file for KERNIX mode |

### Choosing the fingerspelling model (v1 / v2 / v2.1)

In a meeting, each user picks the model in the call toolbar, next to the caption toggle. Switching during a call reconnects recognition with the new model, and the choice is saved in the browser. The single-user demo (`/recognition`) always uses v1. All models are loaded at startup and run on the same `MODEL_RUNTIME`. Letters are confirmed with the same rule as the training project's local webcam test (`webcam_test_model_1.py`, `LetterStabilizer`): predictions below the threshold are skipped, and a letter is confirmed when the last 3 recorded predictions are that letter. Holding the same letter confirms it only once. The thresholds are in `MODELS` in `src/model_api_test/main.py`; `tests/test_confirm.py` checks the rule against that project's code.

| | v1 (default) | v2 / v2.1 |
|---|---|---|
| Model file | `multi_hand_gesture_classifier.tflite` / `KNX_PATH` | `multi_hand_gesture_classifier_v2.{tflite,knx}` / `_v2_1.{tflite,knx}` |
| Input | 10 × 55: joint directions + angles, 0–1 coordinates of the flipped frame | 10 × 75: the same 55 + bone length ratios, pixel coordinates of the unflipped frame |
| Signing hand | Right hand, or the left hand mirrored | The higher of the two hands (left is mirrored) |
| Confirmed when | The same letter 3 times with confidence ≥ 0.9 | The same letter 3 times with confidence ≥ 0.8 |

v2 is the fingerspelling model deployed on 2026-11-17 in [I_CAN_SEE_YOUR_VOICE](https://github.com/yeseoung/I_CAN_SEE_YOUR_VOICE) (local folder `hearing_impaired_helper_make_model`) (see "지문자 v2" in its `PROGRESS.md`). On unseen signers (AI Hub CROWD 18, 19) its letter error rate is 17.9% versus 75.5% for the earlier model. That was measured on OpenPose keypoints, not on a MediaPipe webcam. v2.1 (deployed there on 2026-11-19) has the same architecture, features and settings as v2, trained with 70 extra self-recorded clips; its error rate on the same signers is 17.4%. `tests/test_features.py` checks that the server computes the same features as that project, and `tests/test_hands.py` runs the browser's hand detector (HandLandmarker, as in `static/hands.js`) on that project's letter videos to check that the signing hand reaches the server unmirrored and is recognized.

## API

### HTTP

| Method | Path | Body | Response |
|---|---|---|---|
| `POST` | `/api/login` | `{"username", "password"}` | `{"ok": true, "name", "role", "token"}` or `401` |
| `POST` | `/api/logout` | | `{"ok": true}` |
| `POST` | `/api/meetings` | | `{"code", "host"}` (host is the signed-in user) |
| `GET` | `/api/meetings` | | `{"meetings": [{"code", "host", "participants", "live"}]}` |
| `GET` | `/api/meetings/{code}` | | `{"exists": true, "host"}` or `404` |

Every endpoint except `/api/login` requires `Authorization: Bearer <token>`, using the token from `/api/login`. Without a valid token it returns `401`. Each login creates a new token, and `/api/logout` invalidates it.

### WebSocket `/ws/sign`: recognition

The browser detects hands itself with MediaPipe (`static/hands.js`, `@mediapipe/tasks-vision@1.1.1`, model served from `/models/hand_landmarker.task`). Pick the model with `?model=v1` (default), `?model=v2` or `?model=v2.1`; any other value is refused. For each webcam frame the browser sends one **JSON** message with the 21 hand landmarks as `[x, y]` (0–1, measured on the horizontally flipped frame), or `null` when that hand is not visible, plus the size of that frame (`w`, `h`, required for v2 and v2.1):

```json
{"right": [[0.51, 0.72], [0.48, 0.65], ...], "left": null, "w": 640, "h": 360}
```

The server picks the signing hand as in the table above. Each message gets one JSON reply:

```json
{"detected": true, "action": "ㄱ", "confidence": 0.97, "stable": true, "new": "ㄱ", "frames_collected": 10, "frame_id": 42}
```

- `action` is the last confirmed letter, or `"?"` when the latest prediction is below the threshold. `stable` is true while the last 3 recorded predictions agree.
- `new` is the letter newly confirmed by this frame, or `null`. It is not repeated while the same letter is held. The meeting app adds only `new` to the caption.
- Each connection gets its own 10-frame window, so several people can sign at once without mixing results. The window is cleared when the hand disappears; the confirmation history is kept, as in the local webcam test.
- A message that is not valid JSON, a hand that is not 21 finite `[x, y]` pairs, or (v2) a missing or non-positive `w`/`h` closes the connection with code `1003`.

### WebSocket `/ws/room/{code}?token=<token>`: meeting signaling

The server takes your display name from the session token. It refuses the connection if the token is invalid or the meeting code does not exist. All messages are JSON.

| Direction | `type` | Fields |
|---|---|---|
| server → client | `welcome` | `peerId` (yours), `peers: [{peerId, name}]` |
| server → client | `peer-joined` / `peer-left` | `peerId`, `name` |
| client → server → peer | `signal` | client sends `to`, `data`; peer receives `from`, `data` (WebRTC offer/answer/ICE) |
| client → server → others | `caption` | client sends `text`; others receive `from`, `name`, `text` |

## Project structure

```
src/model_api_test/
├── main.py                 # FastAPI app: pages, REST API, WebSockets, recognition pipeline
├── modules/
│   ├── holistic_module.py  # MediaPipe HandLandmarker wrapper (used by tests/webcam_compare.py)
│   └── utils.py            # Landmarks → joint vectors and angles (model input features)
├── models/
│   ├── hand_landmarker.task                  # MediaPipe hand model
│   ├── multi_hand_gesture_classifier.tflite  # Fingerspelling classifier (default)
│   ├── multi_hand_gesture_classifier_old.knx # The same classifier for KERNIX mode
│   ├── multi_hand_gesture_classifier.knx     # Classifier retrained on 2026-11-05 (KNX_PATH)
│   ├── multi_hand_gesture_classifier_v2.*    # v2 classifier (2026-11-17), .tflite and .knx
│   └── multi_hand_gesture_classifier_v2_1.*  # v2.1 classifier (2026-11-19), .tflite and .knx
└── static/
    ├── hands.js            # In-browser hand landmark detection shared by both pages
    ├── app.html            # Meeting app page: styles and import map (React from esm.sh, no build step)
    ├── app.js              # Meeting app in React + htm (login, meeting list, WebRTC, captions)
    ├── hangul.js           # Jamo-to-syllable composer; self-check: `node src/model_api_test/static/hangul.js`
    └── index.html          # Single-user recognition demo
```

## Known limitations

- **Demo only:** fixed accounts with plain-text passwords in the source. Any signed-in user can join a meeting with its code.
- **In-memory state:** sessions and meetings disappear when the server restarts. Signed-in tabs are sent back to the login screen.
- **Token in the URL:** the meeting WebSocket passes the session token as a query parameter, so it shows up in the server's access log.
- **Shared model lock:** frames from all users go through one model and lock, so many people signing at once slows recognition.
- **Fingerspelling only:** the browser combines recognized letters into Hangul syllables. Word-level signs are not supported.

## Related repositories

- [I_CAN_SEE_YOUR_VOICE](https://github.com/yeseoung/I_CAN_SEE_YOUR_VOICE): model training and data processing
- [KERNIX](https://github.com/yeseoung/KERNIX): Apple Silicon inference runtime

## License

All rights reserved. This code is published for viewing only. Using, copying, modifying, or distributing it requires written permission. See [LICENSE](LICENSE).
