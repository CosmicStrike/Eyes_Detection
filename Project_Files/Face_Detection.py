import cv2

face_Detect = cv2.CascadeClassifier('haarcascade_FaceDetection.xml')
cap = cv2.VideoCapture(0)

cap.set(3, 1024)
cap.set(4, 940)

while True:
    isValid, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    frame = face_Detect.detectMultiScale(imgGray, 1.1, 3)
    # print(len(frame))

    for x, y, w, h in frame:
        cv2.rectangle(img, (x, y), (x+w, y+h), (180, 0, 0), 3)
        # for i in range(x, x+w):
        #     print(i)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
