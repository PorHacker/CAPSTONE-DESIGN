import cv2
import mediapipe as mp
import numpy as np
import time
import os

action = '시험'
idx = 4
seq_length = 30
secs_for_action = 30

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# input video
input_video_dir = './dataset/video/시험/'  # Update this to your directory containing video files

# Create a directory to save the dataset
os.makedirs('dataset', exist_ok=True)

for input_video_file in os.listdir(input_video_dir):
    if input_video_file.endswith(".mp4"):
        input_video_path = os.path.join(input_video_dir, input_video_file)
        video_filename = os.path.splitext(os.path.basename(input_video_path))[0]  # Extracting filename without extension

        # Open the input video
        cap = cv2.VideoCapture(input_video_path)

        data = []

        start_time = time.time()

        # Collect data for a specific action for the specified duration
        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            if not ret:
                break

            # Convert image to RGB for processing
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the image using the hands detection model
            result = hands.process(img)

            # Convert image back to BGR for visualization
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
                    v = v2 - v1  # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                    angle = np.degrees(angle)  # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    d = np.concatenate([joint.flatten(), angle_label])

                    # Append the collected data to the list
                    data.append(d)

                    # Visualize the hand landmarks on the image
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            # Display the image with hand landmarks
            cv2.imshow('img', img)

            # Press 'q' to break out of the loop
            if cv2.waitKey(1) == ord('q'):
                break
        # Close the video capture
        cap.release()

        # Convert collected data to a NumPy array
        data = np.array(data)

        # Print the shape of the collected data
        print(action, data.shape)

        # Save the raw data
        np.save(os.path.join('dataset/raw', f'raw_{action}_{video_filename}'), data)

        # Create sequence data by sliding a window over the collected data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        # Convert sequence data to a NumPy array
        full_seq_data = np.array(full_seq_data)

        # Print the shape of the sequence data
        print(action, full_seq_data.shape)

        # Save the sequence data
        np.save(os.path.join('dataset/seq', f'seq_{action}_{video_filename}'), full_seq_data)

        # Calculate the minimum and maximum values for each feature (excluding the label)
        min_values = np.min(full_seq_data[:, :, :-1], axis=(0, 1))
        max_values = np.max(full_seq_data[:, :, :-1], axis=(0, 1))

        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10

        max_values = np.where(max_values == min_values, max_values + epsilon, max_values)

        # Normalize the sequence data for each feature (X, Y, Z, visibility, and angle)
        normalized_data = (full_seq_data[:, :, :-1] - min_values) / (max_values - min_values)

        # Combine the normalized sequence data with labels
        normalized_data = np.concatenate([normalized_data, full_seq_data[:, :, -1:]], axis=-1)

        # Save the normalized sequence data
        np.save(os.path.join('dataset/normalized', f'normalized_{action}_{video_filename}'), normalized_data)


        cap.release()
        cv2.destroyAllWindows()

