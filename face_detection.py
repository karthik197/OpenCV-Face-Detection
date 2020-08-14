import cv2

# Trained XML file for detecting faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Trained XML file for detecting eyes
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

width = 640
height = 480

#webcam video capture
video = cv2.VideoCapture(0)
video.set(3,width)
video.set(4,height)
while True:
    #reads frame by frame
    ret, img = video.read()

    #grayscale image
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    #faces=face_cascade.detectMultiScale(gray_img,scaleFactor,min_neighbours)
    faces = face_cascade.detectMultiScale(gray_img,1.1,None)

    # To draw a rectangle in face
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        face_gray = gray_img[y: y + h, x: x + w]
        face_color = img[y:y + h, x:x + w]

        # Detects eyes of different sizes in the input image
        eyes = eye_cascade.detectMultiScale(face_gray,1.2,None)

        # To draw a rectangle in eyes
        for (eye_x, eye_y, eye_w, eye_h) in eyes:
            cv2.rectangle(face_color, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 127, 255), 2)

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()