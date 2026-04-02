import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

# Open the first and second camera (device indices 0 and 1)
# Adjust the indices if your cameras are different (e.g., 1 and 2)
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

# Check if both cameras opened successfully
if not cap1.isOpened():
    print("Error: Could not open camera 1 (index 0)")
    exit()
if not cap2.isOpened():
    print("Error: Could not open camera 2 (index 1)")
    exit()


# Get the pose estimation from MediaPipe
with mp_pose.Pose(static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while True:
        # Read a frame from each camera
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Check if frames were read successfully
        if not ret1 or not ret2:
            print("Error: Could not read frames from one or both cameras")
            break

        # Optional: Resize frames to ensure they have the same height
        # This is necessary for hconcat to work if resolutions differ
        height, width, _ = frame1.shape
        frame2_resized = cv2.resize(frame2, (width, height), interpolation=cv2.INTER_AREA)

        # Concatenate the frames horizontally (side by side)
        combined_frame = cv2.hconcat([frame1, frame2_resized])

        # Convert the OpenCV BGR image to RGB format, which MediaPipe requires
        # combined_frame.flags.writeable = False
        combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)

        # Process the image and get pose landmarks
        results = pose.process(combined_frame)

        # Draw the pose annotations on the image
        # combined_frame.flags.writeable = True
        combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(combined_frame,
                                  results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_utils=mp_drawing_styles.get_default_pose_landmarks_style())

        # Display the combined frame
        cv2.imshow('Side by Side Cameras', combined_frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture objects and destroy all windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()