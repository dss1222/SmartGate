from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from gpiozero import Servo
from pygame import mixer
import RPi.GPIO as GPIO
import pigpio
import numpy as np
import cv2
import time
import board
import adafruit_mlx90614
import busio as io
import math

# 텐서플로우, 케라스를 사용하기 위한 라이브러리부터 서보모터, 온도센서, 카메라모듈, 스피커등 센서들을 사용하기 위한 라이브러리
# facenet:얼굴을 찾는 모델
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
# model:마스크검출모델
model = load_model('8LBMI2.h5')
# io포트 사용을 위한 선언
pi = pigpio.pi()
# 실시간 웹캠 읽기
cap = cv2.VideoCapture(0)
i = 0
while cap.isOpened():

    ret, img = cap.read()
    if not ret:
        break
    # 이미지의 높이와 너비 추출
    h, w = img.shape[:2]
    # 이미지 전처리
    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(100, 100), mean=(104., 177., 123.))
    # facent의 input을 blob으로 설정
    facenet.setInput(blob)
    # facent 결과 추론, 얼굴 추출 결과를 dets에 저장
    dets = facenet.forward()

    # 마스크를 착용했는지 확인
    for i in range(dets.shape[2]):
        # 검출한 결과를 신뢰도로 지정
        confidence = dets[0, 0, i, 2]
        # 신뢰도를 0.5임계치로 지정
        if confidence < 0.5:
            continue
        # 바운딩 박스를 구함
        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)
        # 원본 이미지에서 얼굴영역 추출
        face = img[y1:y2, x1:x2]
        face = face / 256

        if (x2 >= w or y2 >= h):
            continue
        if (x1 <= 0 or y1 <= 0):
            continue
        # 추출한 얼굴영역을 전처리
        face_input = cv2.resize(face, (200, 200))
        face_input = np.expand_dims(face_input, axis=0)
        face_input = np.array(face_input)

        # 마스크 검출 모델로 결과값 반환
        modelpredict = model.predict(face_input)
        mask = modelpredict[0][0]
        nomask = modelpredict[0][1]

# 온도센서사용
i2c = io.I2C(board.SCL, board.SDA, frequency=1000000)
mlx = adafruit_mlx90614.MLX90614(i2c)
obj_temp = format(mlx.object_temperature + 2.3, '2f')
amb_temp = format(mlx.ambient_temperature, '2f')
obj = float(obj_temp)
# 시간영역 설정
ms = time.time()
intms = int(ms)

# 마스크 착용유무에 따른 각각의 if설정
if mask > nomask:  # 마스크 착용이 확인될 경우
    color = (0, 255, 0)
    label = 'Mask %d%%' % (mask * 100)
    if obj > 37.5:  # 온도가 37.5도 이상일 경우(약 50cm거리에서 37.5도->35.5도로측정)
        color = (0, 0, 255)
        if intms % 5 == 0:
            mixer.init()
            mixer.music.load('/home/pi/Desktop/ht.mp3')
            mixer.music.play()
            pi.set_servo_pulsewidth(18, 1500)
    elif obj < 36:  # 온도가 제대로 측정되지 않을경우
        color = (0, 0, 255)
        if intms % 5 == 0:
            mixer.init()
            mixer.music.load('/home/pi/Desktop/lt.mp3')
            mixer.music.play()
            pi.set_servo_pulsewidth(18, 1500)
    else:  # 마스크착용과 정상온도가 확인될경우
        color = (0, 255, 0)
        if intms % 5 == 0:
            mixer.init()
            mixer.music.load('/home/pi/Desktop/true.mp3')
            mixer.music.play()
            pi.set_servo_pulsewidth(18, 600)
else:  # 마스크 착용이 확인되지 않을경우
    color = (0, 0, 255)
    label = 'No Mask %d%%' % (nomask * 100)
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    if intms % 5 == 0:
        mixer.init()
        mixer.music.load('/home/pi/Desktop/fales.mp3')
        mixer.music.play()
        pi.set_servo_pulsewidth(18, 1500)

# 화면에 얼굴부분과 마스크 유무를 출력해해줌
txt = str(obj)
tct = label + txt
cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
cv2.putText(img, text=tct, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
            color=color, thickness=2, lineType=cv2.LINE_AA)

cv2.imshow('masktest', img)

if cv2.waitKey(1) & 0xFF == ord('q'):
    break
