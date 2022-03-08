from unittest import skip
import numpy as np
from numpy import linalg as LA
from scipy import fftpack
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image as im
from numpy.linalg import inv

class AR_Detect:
    def __init__(self):
        self.result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (1920, 1080))

    def gethomographyMat(self, cornerList, desiredCoordinates,img1):
        A = np.array([[cornerList[0][0], cornerList[0][1], 1, 0, 0, 0, -cornerList[0][0]*desiredCoordinates[0][0], -cornerList[0][1]*desiredCoordinates[0][0], -desiredCoordinates[0][0]], 
                      [0, 0, 0, cornerList[0][0], cornerList[0][1], 1, -cornerList[0][0]*desiredCoordinates[0][1], -cornerList[0][1]*desiredCoordinates[0][1], -desiredCoordinates[0][1]],

                      [cornerList[1][0], cornerList[1][1], 1, 0, 0, 0, -cornerList[1][0]*desiredCoordinates[1][0], -cornerList[1][1]*desiredCoordinates[1][0], -desiredCoordinates[1][0]],
                      [0, 0, 0, cornerList[1][0], cornerList[1][1], 1, -cornerList[1][0]*desiredCoordinates[1][0], -cornerList[1][1]*desiredCoordinates[1][1], -desiredCoordinates[1][1]],
                      
                      [cornerList[2][0], cornerList[2][1], 1, 0, 0, 0, -cornerList[2][0]*desiredCoordinates[2][0], -cornerList[2][1]*desiredCoordinates[2][0], -desiredCoordinates[2][0]],
                      [0, 0, 0, cornerList[2][0], cornerList[2][1], 1, -cornerList[2][0]*desiredCoordinates[2][0], -cornerList[2][1]*desiredCoordinates[2][1], -desiredCoordinates[2][1]],
                      
                      [cornerList[3][0], cornerList[3][1], 1, 0, 0, 0, -cornerList[3][0]*desiredCoordinates[3][0], -cornerList[3][1]*desiredCoordinates[3][0], -desiredCoordinates[3][0]],
                      [0, 0, 0, cornerList[3][0], cornerList[3][1], 1, -cornerList[3][0]*desiredCoordinates[3][0], -cornerList[3][1]*desiredCoordinates[3][1], -desiredCoordinates[3][1]],
                      
                      ])
        u, s, vh = np.linalg.svd(A)
        homographyMat = np.reshape(vh[8], (3, 3))
       # print(A.dot(vh.T))
        self.pMatrix(homographyMat, desiredCoordinates, cornerList, img1)
        
    def pMatrix(self, homographyMat, cornerList, c, img1):
        K = np.array([[1346.100595, 0, 932.1633975],
                     [0, 1355.933136, 654.8986796],
                     [0, 0, 1]])
        # print(homographyMat)
        homographyMat = inv(homographyMat)
        # print(homographyMat[:, 0])
        lambda_ = 1/((np.linalg.norm(np.matmul(inv(K), homographyMat[:, 0])) + np.linalg.norm(np.matmul(inv(K),homographyMat[:, 1]))/2))
        B_dash = (inv(K).dot(homographyMat))
        
       # print(lambda_)
        if np.linalg.det(B_dash) < 0:
            B_dash = -B_dash
        B = lambda_*B_dash
        r1 = B[:,0]
        r2 = B[:,1]
        r3 = np.cross(r1, r2)
        t = B[:,2]
        R = np.array([r1, r2, r3, t]).T
        P = K.dot(R)
        P1 =   P.dot(np.array([[cornerList[0][0]], [cornerList[0][1]], [-200], [1]]))
        P2 =   P.dot(np.array([[cornerList[1][0]], [cornerList[1][1]], [-200], [1]]))
        P3 =   P.dot(np.array([[cornerList[2][0]], [cornerList[2][1]], [-200], [1]]))
        P4 =   P.dot(np.array([[cornerList[3][0]], [cornerList[3][1]], [-200], [1]]))
        
        P1_x = P1[0]/P1[2]
        P1_y = P1[1]/P1[2]

        P2_x = P2[0]/P2[2]
        P2_y = P2[1]/P2[2]
        
        P3_x = P3[0]/P3[2]
        P3_y = P3[1]/P3[2]
        
        P4_x = P4[0]/P4[2]
        P4_y = P4[1]/P4[2]
    
        color = (0, 255, 0)
  
        # Line thickness of 9 px
        thickness = 9
        img1 = cv2.line(img1, (P1_x, P1_y), (P2_x, P2_y), color, thickness)
        img1 = cv2.line(img1, (P1_x, P1_y), (P3_x, P3_y), color, thickness)
        img1 = cv2.line(img1, (P3_x, P3_y), (P4_x, P4_y), color, thickness)
        img1 = cv2.line(img1, (P2_x, P2_y), (P4_x, P4_y), color, thickness)

        img1 = cv2.line(img1, (P1_x, P1_y), c[0], color, thickness)
        img1 = cv2.line(img1, (P2_x, P2_y), c[1], color, thickness)
        img1 = cv2.line(img1, (P3_x, P3_y), c[2], color, thickness)
        img1 = cv2.line(img1, (P4_x, P4_y), c[3], color, thickness)
        
        img1 = cv2.line(img1, c[1], c[0], color, thickness)
        img1 = cv2.line(img1, c[3], c[1], color, thickness)
        img1 = cv2.line(img1, c[3], c[2], color, thickness)
        img1 = cv2.line(img1, c[0], c[2], color, thickness)
        self.result.write(img1)
       
        cv2.imshow("frame", img1)
        if cv2.waitKey(1) & 0xff == ord('q'):
            cv2.destroyAllWindows()

    def cornerDetector(self):
        cap = cv2.VideoCapture('1tagvideo.mp4')
   
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video  file")
        
        # Read until video is completed
        while(cap.isOpened()):
            ret, img = cap.read()
            if ret == True:
                #img = cv2.imread('frame2.jpg')
                img1 = img
                gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 15, 75, 75)
                ret,img = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
                img = cv2.medianBlur(img,5)
                kernel = np.ones((5,5), np.uint8)
                img = cv2.dilate(img, kernel, iterations=2)
                gray = np.float32(gray)
                
                h, w = img.shape[:2]
                mask = np.zeros((h+2, w+2), np.uint8)
                cv2.floodFill(img, mask, (0,0), 255)
                
                corners = cv2.goodFeaturesToTrack(img,25,0.01,10)
                corners = np.int0(corners)
                
                ext0 = tuple(corners[corners[:, :, 1].argmin()][0])
                ext1 = tuple(corners[corners[:, :, 1].argmax()][0])
                ext2 = tuple(corners[corners[:, :, 0].argmin()][0])
                ext3 = tuple(corners[corners[:, :, 0].argmax()][0])

                y_min = tuple(corners[corners[:, :, 1].argmin()][0])[1]
                y_max = tuple(corners[corners[:, :, 1].argmax()][0])[1]
                x_min = tuple(corners[corners[:, :, 0].argmin()][0])[0]
                x_max = tuple(corners[corners[:, :, 0].argmax()][0])[0]

                color = (0, 255, 0)
                img1 = cv2.circle(img1, ext0, 5, color, -1)
                img1 = cv2.circle(img1, ext1, 5, color, -1)
                img1 = cv2.circle(img1, ext2, 5, color, -1)
                img1 = cv2.circle(img1, ext3, 5, color, -1)
                
                cornerList = [ext0, ext3, ext2, ext1]
                if  100 < x_max - x_min < 200 and 100 < y_max - y_min < 200: 
                    continue  
                desiredCoordinates = [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
                self.gethomographyMat(cornerList, desiredCoordinates, img1)

        
p = AR_Detect()
p.cornerDetector()