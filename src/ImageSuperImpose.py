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
        self.result = cv2.VideoWriter('TerpAR.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (1920, 1080))

    def gethomographyMat(self, cornerList, desiredCoordinates,image_height, image_width, img, terpImg):
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
        self.transformImage(homographyMat, img, desiredCoordinates,image_height, image_width, terpImg)

    def findDistance(self,x1, y1, x2, y2):
        return math.pow(math.pow(x1-x2,2) + math.pow(y1-y2,2), 0.5)

    def transformImage(self, homographyMat, img, desiredCoordinates, image_height, image_width, terpImg):
        homographyInv = inv(homographyMat)
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        ret,gray = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        
        newMat = np.ones([desiredCoordinates[1][0] - desiredCoordinates[0][0],desiredCoordinates[2][1] - desiredCoordinates[1][1],3],dtype=np.uint8)
        cur_x = 0
        cur_y = 0
        dict_ = {}
        for i in range(desiredCoordinates[1][1], desiredCoordinates[2][1]):
            cur_x=0
            for j in range(desiredCoordinates[0][0], desiredCoordinates[1][0]):
                curCord = np.array([[j],[i],[1]])
                newCord = homographyInv.dot(curCord)
                newCord[0] = newCord[0]/newCord[2]
                newCord[1] = newCord[1]/newCord[2]
                if 0 <= int(newCord[1]) < 1080 and 0 <= int(newCord[0]) < 1920:
                    newMat[cur_x, cur_y] = gray[int(newCord[1])][int(newCord[0])]
                    dict_[(cur_x, cur_y)] = curCord
                    cur_x += 1 
            cur_y += 1
        terpImg, skipFrame = self.detectArucoOrientation(newMat, terpImg)
        if not skipFrame:
            for i in range(int(terpImg.shape[:2][1])):
                for j in range(int(terpImg.shape[:2][0])):
                    if (j,i) in dict_:
                        x = dict_[(j, i)][0]
                        y = dict_[(j, i)][1]
                        curCord = np.array([[x],[y],[1]], dtype=object)
                        newCord = homographyInv.dot(curCord)
                        newCord[0] = newCord[0]/newCord[2]
                        newCord[1] = newCord[1]/newCord[2]
                        if 0 <= int(newCord[1]) < 1080 and 0 <= int(newCord[0]) < 1920:
                            img[int(newCord[1])][int(newCord[0])] = terpImg[j, i]
        
            dict_.clear()
            #self.result.write(img)
            cv2.imshow("frame", img)
            if cv2.waitKey(1) & 0xff == ord('q'):
                cv2.destroyAllWindows()

    def detectArucoOrientation(self, newMat, terpImg):
        newMat = cv2.flip(newMat, 0)
        arucoId = 0
        valid_counter = 0
        skipFrame = False
        while arucoId != 13:
            valid_counter += 1
            x, y = newMat.shape[:2]
            x, y = int(x/8), int(y/8)
            binary = []
            binary.append(1 if newMat[int(3*y + y/2), int(3*x + x/2)][0] == 255 else 0)
            binary.append(1 if newMat[int(4*y + y/2), int(3*x + x/2)][0] == 255 else 0)
            binary.append(1 if newMat[int(4*y + y/2), int(4*x + x/2)][0] == 255 else 0)
            binary.append(1 if newMat[int(3*y + y/2), int(4*x + x/2)][0] == 255 else 0)
            arucoId = binary[0]*math.pow(2,0) + binary[1]*math.pow(2,1) + binary[2]*math.pow(2,2) + binary[3]*math.pow(2,3)
            if arucoId == 13:
                print("Image oriented, arucoId: ", arucoId)
                break
            if valid_counter >= 4:
                skipFrame = True
                return terpImg, skipFrame
            newMat = cv2.rotate(newMat, cv2.ROTATE_90_CLOCKWISE)
            terpImg = cv2.rotate(terpImg, cv2.ROTATE_90_CLOCKWISE)
            binary.clear()
      
        terpImg = cv2.resize(terpImg, newMat.shape[:2], interpolation = cv2.INTER_AREA)
        return terpImg, skipFrame
        
        # print("exited")
        # plt.imshow(newMat, plt.cm.gray)   
        # plt.show()  
        
    def ARTag(self):
        cap = cv2.VideoCapture('1tagvideo.mp4')
   
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video  file")
        
        # Read until video is completed
        while(cap.isOpened()):
            ret, img = cap.read()
            if ret == True:
                #img = cv2.imread('frame15.jpg')
                terpImg = cv2.imread('testudo.png')
                img1 = img
                gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
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
        
                # img1 = cv2.circle(img1, ext0, 5, (255,0, 0), -1)
                # img1 = cv2.circle(img1, ext1, 5, (255,0, 0), -1)
                # img1 = cv2.circle(img1, ext2, 5, (255,0, 0), -1)
                # img1 = cv2.circle(img1, ext3, 5, (255,0, 0), -1)
                
                cornerList = [ext0, ext3, ext2, ext1]
                image_width, image_height =  np.abs(x_min - x_max), np.abs(y_min -y_max)
                desiredCoordinates = [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
                self.gethomographyMat(cornerList, desiredCoordinates,image_height, image_width, img1, terpImg)
      
p = AR_Detect()
p.ARTag()