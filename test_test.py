import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
import time
from collections import Counter
from collections import OrderedDict
import os

actions = ['나', '개학', '홍익대학교', '가다', '시험', '복습', '시작']
seq_length = 30

# Set the Korean font file path
font_path = "D:/Python 3.7.8/Lib/site-packages/matplotlib/mpl-data/fonts/NanumGothic.ttf"
# 경로를 실제 파일 위치로 변경

# Load Korean fonts.
font_size = 80
font = ImageFont.truetype(font_path, font_size)
font_color = (255, 255, 255)
font_thickness = 2

model = load_model('models/model_r8.h5')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# Set the webcam's width and height
cap.set(3, 1920)  # Width to 1920 pixels
cap.set(4, 1080)  # Height to 1080 pixels

seq = []
action_seq = []
last_save_time = time.time()  # 초기 저장 시간 설정
recognized_words = []
current_word = None  # 현재 인식된 단어
arr=[]

# 웹캠이 시작할 때 gesture_results.txt 파일 초기화
with open('gesture_results.txt', 'w') as f:
    f.write('')

while cap.isOpened():
    ret, img = cap.read()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
            angle = np.degrees(angle)

            d = np.concatenate([joint.flatten(), angle])
            seq.append(d)
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.95:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?'
            if action_seq[-1]:
                this_action = action

            if this_action != current_word:
                current_word = this_action
                recognized_words.append(current_word)

            # 한글 텍스트를 이미지로 렌더링
            if current_word in actions:
                pil_img = Image.fromarray(img)
                draw = ImageDraw.Draw(pil_img)
                text_size = draw.textsize(current_word, font=font)
                x = int(res.landmark[0].x * img.shape[1])
                y = int(res.landmark[0].y * img.shape[0] + 20)
                # draw.text((x, y), current_word, font=font, fill=font_color)
                img = np.array(pil_img)

    cv2.imshow('Gesture Recognition', img)

    # 3초마다 결과값을 텍스트 파일에 저장
    current_time = time.time()
    if current_time - last_save_time >= 3:
        last_save_time = current_time
        if recognized_words:
            most_common_word = Counter(recognized_words).most_common(1)[0][0]
            arr.append(most_common_word)

            with open('gesture_results.txt', 'w') as f:
                arr_items = list(OrderedDict.fromkeys(arr))
                f.write(' '.join(arr_items) + '\n')
            recognized_words = []

    # gesture_results.txt 파일 불러오기
    with open('gesture_results.txt', 'r') as f:
        content = f.read()

    # 한글 텍스트를 이미지로 렌더링
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    text_size = draw.textsize(content, font=font)
    x = int((img.shape[1] - text_size[0]) // 3) + 100  # 이미지 중앙보다 조금 왼쪽으로
    y = (img.shape[0] // 2) + 140  # 이미지 중앙에서 100 픽셀 아래에 출력
    draw.text((x, y), content, font=font, fill=(255, 255, 255))  # 흰색으로 출력
    img = np.array(pil_img)

    # 결과를 실시간으로 표시
    cv2.imshow('Gesture Recognition', img)

    # cv2.imshow와 cv2.waitKey를 이용하여 화면을 유지
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()