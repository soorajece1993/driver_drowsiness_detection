# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 16:54:54 2018

@author: sharan.sailajadevi ok
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 11:54:22 2018

@author: sharan.sailajadevi
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from keras.models import load_model
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
leye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
botheye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def convrtToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def cnnPreprocess(img):
	img = img.astype('float32')
	img /= 255
	img = np.expand_dims(img, axis=2)
	img = np.expand_dims(img, axis=0)
	return img
def detect_faces(f_cascade,colored_img,scaleFactor = 1.2):    
    img_copy = np.copy(colored_img)
    #mg_copy = colored_img
    test2 = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    #clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    #gray = clahe.apply(test2)
    gray = cv2.equalizeHist(test2) 
    faces = haar_face_cascade.detectMultiScale(gray,scaleFactor=scaleFactor, minNeighbors=3);
    #if faces is not None:
    print("length")
    print(len(faces))
    if len(faces) == 0:
        print("CLAHE") 
        haar1_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        test2 = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        gray = clahe.apply(test2)
        faces = haar1_face_cascade.detectMultiScale(gray,scaleFactor=scaleFactor, minNeighbors=3);
        if len(faces) != 0:
            print("second cascade")
            for (x, y, w, h) in faces:
               cv2.rectangle(img_copy, (x,y), (x+w, y+h), (255, 0, 0),2)
               img_gray = gray[y:y+h, x:x+w]
               img_color = img_copy[y:y+h, x:x+w]
#        cv2.imwrite("D:/mproj/grayim"+".jpg", img_gray)
#    n+=1    
            return img_copy,img_gray,img_color,'T'
        else:
            return None,None,None,'F'
    else:
        for (x, y, w, h) in faces:
            print("first face cascade")
            cv2.rectangle(img_copy, (x,y), (x+w, y+h), (255, 0, 0),2)
            img_gray = gray[y:y+h, x:x+w]
            img_color = img_copy[y:y+h, x:x+w]
            break
#        cv2.imwrite("D:/mproj/grayim"+".jpg", img_gray)
#    n+=1    
        return img_copy,img_gray,img_color,'T'

def detect_eyes(i_leye_cascade, ii_gray_img, i_col_img):
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    i_gray_img = clahe.apply(ii_gray_img)
    leyes = leye_cascade.detectMultiScale(i_gray_img)
    print("first")
    if len(leyes) != 0:    
        for (ex1,ey1,ew1,eh1) in leyes:
            eye1c=i_col_img[ey1:ey1+eh1, ex1:ex1+ew1]
            resize_imgc = cv2.resize(eye1c  , (24 , 24))
            cv2.rectangle(i_col_img,(ex1,ey1),(ex1+ew1,ey1+eh1),(0,255,0),2)
            eye1=i_gray_img[ey1:ey1+eh1, ex1:ex1+ew1]
            resize_img = cv2.resize(eye1  , (24 , 24))
            return resize_imgc,eye1,resize_img,'T'
#        cv2.imwrite("D:/mproj/eye1"+".jpg", resize_img)
#        cv2.imshow('eye1', eye1)
    reye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
    reyes = reye_cascade.detectMultiScale(i_gray_img)
    print("second")
    if len(reyes) !=0:
        for (ex2,ey2,ew2,eh2) in reyes:
            eye2c=i_col_img[ey2:ey2+eh2, ex2:ex2+ew2]
            resize_imgc = cv2.resize(eye2c  , (24 , 24))
            cv2.rectangle(i_col_img,(ex2,ey2),(ex2+ew2,ey2+eh2),(0,255,0),2)
            eye2=i_gray_img[ey2:ey2+eh2, ex2:ex2+ew2]
            resize_img = cv2.resize(eye2  , (24 , 24))
        return resize_imgc,eye2,resize_img,'T'
#        cv2.imwrite("D:/mproj/eye2"+".jpg", eye2)
#        cv2.imshow('eye2', eye2)
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')       
    eyes = eye_cascade.detectMultiScale(i_gray_img)
    print("third")
    if len(eyes) !=0:
        for (ex3,ey3,ew3,eh3) in eyes:
            eye3c=i_col_img[ey3:ey3+eh3, ex3:ex3+ew3]
            resize_imgc = cv2.resize(eye3c  , (24 , 24))
            cv2.rectangle(i_col_img,(ex3,ey3),(ex3+ew3,ey3+eh3),(0,255,0),2)
            eye3=i_gray_img[ey3:ey3+eh3, ex3:ex3+ew3]
            resize_img = cv2.resize(eye3  , (24 , 24))
        return resize_imgc,eye3,resize_img,'T'
#        cv2.imwrite("D:/mproj/eye3"+".jpg", eye3)
#        cv2.imshow('eye3', eye3)
    botheye_cascade = cv2.CascadeClassifier('haarcascade_mcs_lefteye.xml')    
    beyes = botheye_cascade.detectMultiScale(i_gray_img)
    print("fourth")
    if len(beyes) !=0:
        for (ex4,ey4,ew4,eh4) in beyes:
            eye4c=i_col_img[ey4:ey4+eh4, ex4:ex4+ew4]
            resize_imgc = cv2.resize(eye4c  , (24 , 24))
            cv2.rectangle(i_col_img,(ex4,ey4),(ex4+ew4,ey4+eh4),(0,255,0),2)
            eye4=i_gray_img[ey4:ey4+eh4, ex4:ex4+ew4]
            resize_img = cv2.resize(eye4  , (50 , 50))
           # break
        return resize_imgc,eye4,resize_img,'T'
    else:
       return None,None,None,'F'
    
def main():
    test1=cv2.imread('testnew.jpg')
    #test1=cv2.imread('/home/sharanss2001/mproj/indata/ArcSoft_Image5.jpg')
    faces_detected_img,gray_img,col_img,faceflag = detect_faces(haar_face_cascade, test1)
#    print("face value")
#    print(faces_detected_img)
    if (faceflag != 'F'):
#            cv2.imshow('Test img', faces_detected_img)
#            cv2.waitKey(0)
            eyes_detected_img,gray_i_img,re_img,eyesflag = detect_eyes(leye_cascade, gray_img,col_img)
            if (eyesflag != 'F'):
                
#                plt.imshow((convrtToRGB(faces_detected_img)))
                plt.imshow(faces_detected_img)
#                 plt.imshow(faces_detected_img,cmap=None, interpolation = 'bicubic')
                plt.xticks([]),plt.yticks([])
                plt.show
 #                plt.imshow(re_img)
 #               plt.xticks([]),plt.yticks([])
 #               plt.show
 #               plt.imshow(faces_detected_img, cmap='gray', interpolation = 'bicubic')
 #               plt.xticks([]),plt.yticks([])
 #               plt.show
 #               cv2.imshow('Test img', faces_detected_img)
 #               cv2.waitKey(0)
 #               cv2.imshow('Test eye img', eyes_detected_img)
 #               cv2.waitKey(0)
 #               cv2.imshow('Test eye resize img', re_img)
 #               cv2.waitKey(0)
 #               cv2.destroyAllWindows()
                 #test12=cv2.imread('/home/sharanss2001/mproj/Aaron_Guiel_0001_L.jpg',0)
                 #test12=cv2.imread('/home/sharanss2001/mproj/indata/eye4.jpg',0) 
                 #eyes_detected_img1 = cv2.resize(eyes_detected_img  , (24 , 24))
                grayc = cv2.cvtColor(eyes_detected_img, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
                grayc = clahe.apply(grayc)
                #grayc = cv2.equalizeHist(grayc)
                img2 = np.zeros_like(eyes_detected_img)
                img2[:,:,0] = grayc  
                img2[:,:,1] = grayc
                img2[:,:,2] = grayc
                grayco = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                cv2.imwrite("imagesooraj.jpg",grayco)
                 
                model = load_model('blinkModelv8.hdf5')
                prediction = (model.predict(cnnPreprocess(grayco)))
                print('prediction done')
                print(prediction)
                if prediction > 0.5 :
                    state = 'open'
                    close_counter = 0  
                else:
                    state = 'close'    
        #             close_counter += 1
		
		# if the eyes are open and previousle were closed
		# for sufficient number of frames then increcement 
		# the total blinks
                if state == 'open':
                    print('eyes open')
                else:
                    print('eyes closed')
 
            else:
                print("No eyesdetected")   
    else:   
#        faces_detected_img == None):
        print("No faces detected")
    
        
if __name__ == '__main__':
	main()  
