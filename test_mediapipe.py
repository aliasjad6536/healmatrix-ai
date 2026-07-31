import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
result = pose.process(dummy_image)
print("Pose object created and ran successfully - no crash")
print("Landmarks detected on blank image:", result.pose_landmarks is not None)