
import cv2
import mediapipe as mp
import time

from pathlib import Path
from types import SimpleNamespace


# =========================================================
# Model path
# =========================================================

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "hand_landmarker.task"
)


class HolisticDetector:
    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.refine_face_landmarks = refine_face_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Hand Landmarker model not found: {MODEL_PATH}"
            )

        # MediaPipe Tasks API
        self.vision = mp.tasks.vision

        running_mode = (
            self.vision.RunningMode.IMAGE
            if static_image_mode
            else self.vision.RunningMode.VIDEO
        )

        options = self.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=running_mode,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.hand_detector = (
            self.vision.HandLandmarker.create_from_options(options)
        )

        self.results = SimpleNamespace(
            pose_landmarks=None,
            face_landmarks=None,
            left_hand_landmarks=None,
            right_hand_landmarks=None,
        )

        self.left_hand_lmList = []
        self.right_hand_lmList = []
        self.pose_lmList = []
        self.face_lmList = []

        self.tipIds = [4, 8, 12, 16, 20]
        self._last_timestamp_ms = -1

    # =====================================================
    # Internal helpers
    # =====================================================

    @staticmethod
    def _make_landmark_list(landmarks):
        """
        Wrap Tasks API landmarks in the .landmark interface
        expected by the existing application code.
        """
        if landmarks is None:
            return None

        return SimpleNamespace(landmark=landmarks)

    @staticmethod
    def _pixel_coordinates(img, landmarks):
        points = []
        h, w = img.shape[:2]

        if landmarks is None:
            return points

        for idx, lm in enumerate(landmarks.landmark):
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cz = int(lm.z * (w + h) / 2)

            points.append([idx, cx, cy, cz])

        return points

    def _next_timestamp_ms(self):
        timestamp_ms = time.monotonic_ns() // 1_000_000

        # VIDEO mode requires monotonically increasing timestamps.
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1

        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    # =====================================================
    # Hand detection
    # =====================================================

    def findHolistic(self, img, draw=True):
        """
        Detect hands using MediaPipe Tasks HandLandmarker.

        Note:
            This method no longer detects pose or face landmarks.
            The method name is retained for compatibility.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb,
        )

        if self.static_image_mode:
            detected = self.hand_detector.detect(mp_image)
        else:
            detected = self.hand_detector.detect_for_video(
                mp_image,
                self._next_timestamp_ms(),
            )

        left_hand = None
        right_hand = None

        # Match each landmark set to its handedness classification.
        for index, landmarks in enumerate(detected.hand_landmarks):
            handedness = detected.handedness[index]

            if not handedness:
                continue

            label = handedness[0].category_name

            if label == "Right":
                right_hand = landmarks
            elif label == "Left":
                left_hand = landmarks

        # Preserve the interface used by main.py:
        # results.right_hand_landmarks.landmark
        self.results = SimpleNamespace(
            pose_landmarks=None,
            face_landmarks=None,
            left_hand_landmarks=self._make_landmark_list(left_hand),
            right_hand_landmarks=self._make_landmark_list(right_hand),
        )

        # Update pixel-coordinate lists on every frame.
        self.left_hand_lmList = self._pixel_coordinates(
            img, self.results.left_hand_landmarks
        )
        self.right_hand_lmList = self._pixel_coordinates(
            img, self.results.right_hand_landmarks
        )

        if draw:
            for hand in (left_hand, right_hand):
                if hand is None:
                    continue

                # Draw landmark points without relying on legacy
                # mp.solutions drawing utilities.
                h, w = img.shape[:2]

                for lm in hand:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

        return img

    def findRighthandLandmark(self, img, draw=False):
        """
        Return:
            pixel_landmarks: [[id, x, y, z], ...]
            normalized_landmarks: 21 normalized landmarks or None
        """
        self.right_hand_lmList = self._pixel_coordinates(
            img,
            self.results.right_hand_landmarks,
        )

        return (
            self.right_hand_lmList,
            self.results.right_hand_landmarks,
        )

    def findLefthandLandmark(self, img, draw=True):
        self.left_hand_lmList = self._pixel_coordinates(
            img,
            self.results.left_hand_landmarks,
        )

        return (
            self.left_hand_lmList,
            self.results.left_hand_landmarks,
        )

    def close(self):
        self.hand_detector.close()
