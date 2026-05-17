from imutils import face_utils
import dlib
import cv2
import math

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

def euclidean_distance(x1, y1, x2,y2):
    x_squared = (x1 - x2) ** 2
    y_squared = (y1 - y2) ** 2
    return math.sqrt(x_squared + y_squared)

def EAR(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        
        ##Right eye
        d1 = euclidean_distance(landmarks[43][0], landmarks[43][1], landmarks[47][0], landmarks[47][1])
        d2 = euclidean_distance(landmarks[44][0], landmarks[44][1], landmarks[46][0], landmarks[46][1])
        d3 = 2 * euclidean_distance(landmarks[42][0], landmarks[42][1], landmarks[45][0], landmarks[45][1])
        EAR1 = (d1 + d2)/d3
        
        ##Left eye
        d4 = euclidean_distance(landmarks[37][0], landmarks[37][1], landmarks[41][0], landmarks[41][1])
        d5 = euclidean_distance(landmarks[38][0], landmarks[38][1], landmarks[40][0], landmarks[40][1])
        d6 = 2 * euclidean_distance(landmarks[36][0], landmarks[36][1], landmarks[39][0], landmarks[39][1])
        EAR2 = (d4 + d5)/d6
        EAR_avg = (EAR1 + EAR2)/2
        return EAR_avg
