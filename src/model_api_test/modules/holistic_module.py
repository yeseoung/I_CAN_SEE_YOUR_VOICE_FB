
import cv2
import mediapipe as mp
import math
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

    # =====================================================
    # Pose and face compatibility methods
    # =====================================================

    def findPoseLandmark(self, img, draw=True):
        self.pose_lmList = []
        return self.pose_lmList

    def findFaceLandmark(self, img, draw=True):
        self.face_lmList = []
        return self.face_lmList

    # =====================================================
    # Finger state detection
    # =====================================================

    def left_hand_fingersUp(self, axis=False):
        return self._fingers_up(
            self.left_hand_lmList,
            is_right=False,
            axis=axis,
        )

    def right_hand_fingersUp(self, axis=False):
        return self._fingers_up(
            self.right_hand_lmList,
            is_right=True,
            axis=axis,
        )

    def _fingers_up(self, landmarks, is_right, axis=False):
        if len(landmarks) < 21:
            return []

        fingers = []

        thumb_tip = landmarks[self.tipIds[0]]
        thumb_base = landmarks[self.tipIds[0] - 2]
        little_tip = landmarks[self.tipIds[4]]

        if axis:
            fingers.append(
                int(thumb_tip[2] < thumb_base[2])
            )

            if thumb_tip[1] < little_tip[1]:
                for i in range(1, 5):
                    fingers.append(
                        int(
                            landmarks[self.tipIds[i]][1]
                            > landmarks[self.tipIds[i] - 2][1]
                        )
                    )
            else:
                for i in range(1, 5):
                    fingers.append(
                        int(
                            landmarks[self.tipIds[i]][1]
                            < landmarks[self.tipIds[i] - 2][1]
                        )
                    )

            return fingers

        # Thumb direction depends on the hand.
        if is_right:
            if thumb_tip[1] > little_tip[1]:
                fingers.append(
                    int(thumb_tip[1] > thumb_base[1])
                )
            else:
                fingers.append(
                    int(thumb_tip[1] < thumb_base[1])
                )
        else:
            if thumb_tip[1] < little_tip[1]:
                fingers.append(
                    int(thumb_tip[1] < thumb_base[1])
                )
            else:
                fingers.append(
                    int(thumb_tip[1] > thumb_base[1])
                )

        # Index, middle, ring, and little fingers.
        for i in range(1, 5):
            fingers.append(
                int(
                    landmarks[self.tipIds[i]][2]
                    < landmarks[self.tipIds[i] - 2][2]
                )
            )

        return fingers

    # =====================================================
    # Geometry helpers
    # =====================================================

    def findCenter(self, p1, p2):
        x1, y1 = self.pose_lmList[p1][1:3]
        x2, y2 = self.pose_lmList[p2][1:3]

        return (
            (x1 + x2) // 2,
            (y1 + y2) // 2,
        )

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.face_lmList[p1][1:3]
        x2, y2 = p2[0], p2[1]

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)

        return math.hypot(x2 - x1, y2 - y1), img

    def findDepth(self, p1, p2):
        return abs(
            (
                self.pose_lmList[p1][3]
                + self.pose_lmList[p2][3]
            ) / 2
        )

    def findEyeBlink(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.face_lmList[p1][1:3]
        x2, y2 = self.face_lmList[p2][1:3]

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)

        return math.hypot(x2 - x1, y2 - y1), img

    def findEyeDepth(self, p1, p2):
        return abs(
            (
                self.face_lmList[p1][3]
                + self.face_lmList[p2][3]
            ) / 2
        )

    def drawLine(self, p1, p2, img, t=3):
        x1, y1 = self.face_lmList[p1][1:3]
        x2, y2 = self.face_lmList[p2][1:3]

        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), t)

    def findLength_lh_rh(self, p1, p2):
        x1, y1 = self.left_hand_lmList[p2][1:3]
        x2, y2 = self.right_hand_lmList[p1][1:3]

        return math.hypot(x2 - x1, y2 - y1)

    def findLength_lh_lh(self, p1, p2):
        x1, y1 = self.left_hand_lmList[p2][1:3]
        x2, y2 = self.left_hand_lmList[p1][1:3]

        return math.hypot(x2 - x1, y2 - y1)

    def findLength_rh_rh(self, p1, p2):
        x1, y1 = self.right_hand_lmList[p2][1:3]
        x2, y2 = self.right_hand_lmList[p1][1:3]

        return math.hypot(x2 - x1, y2 - y1)

    def findLength_pose(self, p1, p2):
        x1, y1 = self.pose_lmList[p2][1:3]
        x2, y2 = self.pose_lmList[p1][1:3]

        return math.hypot(x2 - x1, y2 - y1)

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.pose_lmList[p1][1:3]
        x2, y2 = self.pose_lmList[p2][1:3]
        x3, y3 = self.pose_lmList[p3][1:3]

        radian = (
            math.atan2(y3 - y2, x3 - x2)
            - math.atan2(y1 - y2, x1 - x2)
        )
        angle = math.degrees(radian)

        if angle < 0:
            angle += 360

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)

            for x, y in ((x1, y1), (x2, y2), (x3, y3)):
                cv2.circle(img, (x, y), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x, y), 15, (0, 0, 255), 2)

            cv2.putText(
                img,
                str(int(angle)),
                (x2 - 50, y2 + 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                2,
            )

        return angle

    def findHandAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.right_hand_lmList[p1][1:3]
        x2, y2 = self.right_hand_lmList[p2][1:3]
        x3, y3 = self.right_hand_lmList[p3][1:3]

        radian = (
            math.atan2(y3 - y2, x3 - x2)
            - math.atan2(y1 - y2, x1 - x2)
        )
        angle = math.degrees(radian)

        if angle < 0:
            angle += 360

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)

            for x, y in ((x1, y1), (x2, y2), (x3, y3)):
                cv2.circle(img, (x, y), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x, y), 15, (0, 0, 255), 2)

            cv2.putText(
                img,
                str(int(angle)),
                (x2 - 50, y2 + 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                2,
            )

        return angle

    def close(self):
        self.hand_detector.close()
