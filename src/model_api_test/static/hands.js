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
            // 예전 서버처럼 좌우 반전한 프레임에서 검출해야 좌우 손 판정과 좌표가 학습 때와 같다
            ctx.setTransform(-1, 0, 0, 1, W, 0);
            ctx.drawImage(video, 0, 0, W, H);
            lastTs = Math.max(performance.now(), lastTs + 1);  // VIDEO 모드는 시간이 계속 늘어야 한다
            const r = landmarker.detectForVideo(canvas, lastTs);
            const hands = { right: null, left: null, w: W, h: H };
            r.landmarks.forEach((lm, i) => {
                const label = r.handedness[i]?.[0]?.categoryName;
                if (label === "Right" || label === "Left") {
                    hands[label.toLowerCase()] = lm.map(p => [p.x, p.y]);
                }
            });
            return hands;
        },
        close() { landmarker.close(); },
    };
}
