// 브라우저에서 손 랜드마크를 뽑아 /ws/sign에 보낼 메시지를 만든다 (서버 Python mediapipe 1.1.1과 같은 버전)
import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/vision_bundle.mjs";

const W = 640;
const fileset = FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm");

export async function createHandTracker() {
    const landmarker = await HandLandmarker.createFromOptions(await fileset, {
        baseOptions: { modelAssetPath: "/models/hand_landmarker.task" },
        runningMode: "VIDEO",
        numHands: 2,
        // 예전 서버 검출기 설정 그대로 (min_detection_confidence=0.3)
        minHandDetectionConfidence: 0.3,
        minHandPresenceConfidence: 0.3,
        minTrackingConfidence: 0.5,
    });
    const canvas = document.createElement("canvas");
    canvas.width = W;
    const ctx = canvas.getContext("2d");
    let lastTs = -1;

    return {
        // 영상 프레임 → {right, left, w, h}: 손은 각각 21개 [x, y](0~1) 또는 null, w·h는 검출한 프레임 크기
        detect(video) {
            // 카메라의 가로세로 비율을 지킨다: v2 모델은 픽셀 좌표(0~1 × w, h)로 학습했다
            const H = Math.round(W * video.videoHeight / video.videoWidth) || 480;
            if (canvas.height !== H) canvas.height = H;
            // 서버는 좌우 반전한 프레임의 좌표를 받는다 (예전 서버와 같은 형식)
            ctx.setTransform(-1, 0, 0, 1, W, 0);
            ctx.drawImage(video, 0, 0, W, H);
            lastTs = Math.max(performance.now(), lastTs + 1);  // VIDEO 모드는 시간이 계속 늘어야 한다
            const r = landmarker.detectForVideo(canvas, lastTs);
            const hands = { right: null, left: null, w: W, h: H };
            r.landmarks.forEach((lm, i) => {
                const label = r.handedness[i]?.[0]?.categoryName;
                // HandLandmarker는 받은 화면에서 보이는 모양으로 좌우를 붙인다: 반전한 프레임에서 사용자의 오른손은 "Left".
                // 그대로 쓰면 서버가 오른손을 왼손으로 보고 반전해 정답률이 18%로 떨어진다 (tests/test_hands.py)
                if (label === "Right" || label === "Left") {
                    hands[label === "Right" ? "left" : "right"] = lm.map(p => [p.x, p.y]);
                }
            });
            return hands;
        },
        close() { landmarker.close(); },
    };
}
