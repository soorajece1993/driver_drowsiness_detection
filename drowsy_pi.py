import RPi.GPIO as GPIO
from imutils.video import VideoStream
import imutils
 
import time
from keras.models import load_model
import time
import cv2
import numpy as np
import serial

#ser = serial.Serial(port='COM4', baudrate=9600)
#ser.close()
#ser.open()
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(2,GPIO.OUT,initial=GPIO.LOW)

t1=0
t2=0
t3=0
t4=0
slept=0
abtslp=0
noface=0
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
leye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
botheye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# initialize the camera and grab a reference to the raw camera capture
def convrtToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def cnnPreprocess(img):
	img = img.astype('float32')
	img /= 255
	img = np.expand_dims(img, axis=2)
	img = np.expand_dims(img, axis=0)
	return img

def detect_eyes(i_leye_cascade, ii_gray_img, i_col_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    i_gray_img = clahe.apply(ii_gray_img)
    leyes = leye_cascade.detectMultiScale(i_gray_img)
    print("first")
    if len(leyes) != 0:
        for (ex1, ey1, ew1, eh1) in leyes:
            eye1c = i_col_img[ey1:ey1 + eh1, ex1:ex1 + ew1]
            resize_imgc = cv2.resize(eye1c, (24, 24))
            cv2.rectangle(i_col_img, (ex1, ey1), (ex1 + ew1, ey1 + eh1), (0, 255, 0), 2)
            eye1 = i_gray_img[ey1:ey1 + eh1, ex1:ex1 + ew1]
            resize_img = cv2.resize(eye1, (24, 24))
            return resize_imgc, eye1, resize_img, 'T'

    reye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
    reyes = reye_cascade.detectMultiScale(i_gray_img)
    print("second")
    if len(reyes) != 0:
        for (ex2, ey2, ew2, eh2) in reyes:
            eye2c = i_col_img[ey2:ey2 + eh2, ex2:ex2 + ew2]
            resize_imgc = cv2.resize(eye2c, (24, 24))
            cv2.rectangle(i_col_img, (ex2, ey2), (ex2 + ew2, ey2 + eh2), (0, 255, 0), 2)
            eye2 = i_gray_img[ey2:ey2 + eh2, ex2:ex2 + ew2]
            resize_img = cv2.resize(eye2, (24, 24))
        return resize_imgc, eye2, resize_img, 'T'


    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    eyes = eye_cascade.detectMultiScale(i_gray_img)
    print("third")
    if len(eyes) != 0:
        for (ex3, ey3, ew3, eh3) in eyes:
            eye3c = i_col_img[ey3:ey3 + eh3, ex3:ex3 + ew3]
            resize_imgc = cv2.resize(eye3c, (24, 24))
            cv2.rectangle(i_col_img, (ex3, ey3), (ex3 + ew3, ey3 + eh3), (0, 255, 0), 2)
            eye3 = i_gray_img[ey3:ey3 + eh3, ex3:ex3 + ew3]
            resize_img = cv2.resize(eye3, (24, 24))
        return resize_imgc, eye3, resize_img, 'T'


    else:
        return None, None, None, 'F'


model = load_model('blinkModelv8.hdf5')
#cap = cv2.VideoCapture(0)

# Are we using the Pi Camera?
usingPiCamera = True
# Set initial frame size.
frameSize = (320, 240)
 
# Initialize mutithreading the video stream.
vs = VideoStream(src=0, usePiCamera=usingPiCamera, resolution=frameSize,
		framerate=32).start()
# Allow the camera to warm up.
time.sleep(2.0)

while 1:

    frame = vs.read()
    if len(frame)>0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_face_cascade.detectMultiScale(frame, 1.3, 5)
        if len(faces) ==0:
            noface=noface+1
            if noface>10:
                print("Alert..Alert..Sit properly behind the wheels")
                noface=0
        #print (faces)
        else:
            noface=0
            for (x, y, w, h) in faces:
            # To draw a rectangle in a face
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

            # faces_detected_img, gray_img, col_img, faceflag = detect_faces(haar_face_cascade, frame)
            #if len(faces)>0:
                #cv2.imshow('Test img', roi_gray)
                eyes_detected_img, gray_i_img, re_img, eyesflag = detect_eyes(leye_cascade, roi_gray, roi_color)

                if (eyesflag != 'F'):
                    grayc = cv2.cvtColor(eyes_detected_img, cv2.COLOR_BGR2GRAY)
                    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    #grayc = clahe.apply(grayc)
                    grayc = cv2.equalizeHist(grayc)
                    img2 = np.zeros_like(eyes_detected_img)
                    img2[:, :, 0] = grayc
                    img2[:, :, 1] = grayc
                    img2[:, :, 2] = grayc
                    seconds = time.time()
                    result = time.localtime(seconds)
                    result1 = time.localtime(seconds+4)
                    result2 = time.localtime(seconds+50)
                    if (t3==t4):
                        t4=result2.tm_sec
                    t3=result.tm_sec
                    if (t1==t2):
                        print("entering")
                        t2=result1.tm_sec
                    t1=result.tm_sec
                
                    print("seconds",t1)
                    print("seconds+4",t2)
                    grayco = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                          
                    #cv2.imwrite("test.jpg",grayco)
                    prediction = (model.predict(cnnPreprocess(grayco)))
                    #print('prediction done')
                    #print(prediction)
                    if prediction > 0.5 :
                        state = 'open'
                        close_counter = 0
                    else:
                        state = 'close'

                    if state == 'open':
                        print('eyes open')

                        #if t1 == t2:
                        t1=0
                        t2=0
                        slept=0;
                    else:
                        print('eyes closed')
                        #print("value of x ",x)
                        #print("value of y ",y)
                        abtslp=abtslp+1
                        slept=slept+1
                        if t1 == t2:
                            if slept > 3:
                               print("Alert..Wake Up..Wake Up")
                               a='A'
                               a=a.encode()
                               #ser.write(a)
                               GPIO.output(2,GPIO.HIGH)
                               time.sleep(3)
                               GPIO.output(2,GPIO.LOW)
                               

                               print("Alert..Wake Up.. Wake Up")
                               slept =0
                               t1=0
                               t2=0
                            else:
                                slept=0
                                x=0
                                y=0;
                    print("t3:",t3)
                    print("t4:",t4)
                    print("abtslp:",abtslp)
                    if t3==t4:
                        if abtslp<10:
                            print('Alert......You are going to sleep')
                            print('Alert......You are going to sleep')
                            
                            GPIO.output(2,GPIO.HIGH)
                            time.sleep(2)
                            GPIO.output(2,GPIO.LOW)
                            
                            abtslp=0
                            t3=0
                            t4=0;

                        #else:
                            #GPIO.output(buzzer,GPIO.HIGH)
                            
                    else:
                        print("Checking...");
                        
          #  else:
                
           #     print("No faces detected")

        cv2.imshow("frames", frame)
        cv2.waitKey(20)



