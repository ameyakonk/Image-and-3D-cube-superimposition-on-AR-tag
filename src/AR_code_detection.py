import numpy as np
from numpy import linalg as LA
from scipy import fftpack
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image as im
from numpy.linalg import inv

class AR_Detect:

    def readVideo(self):
        cap = cv2.VideoCapture('1tagvideo.mp4')
        if (cap.isOpened()== False):
            print("Error opening video stream or file")
        count = 0
        while(cap.isOpened()):
        # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # Display the resulting frame
                cv2.imshow('Frame',frame)
                #cv2.imwrite("frame%d.jpg" % count, frame)
                count += 1
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else: 
                break
        cap.release()
        cv2.destroyAllWindows()

    def gethomographyMat(self, cornerList, desiredCoordinates, img, terpImg):
        #desiredCoordinates = [[0,0], [terpImg.shape[0], 0], [0, terpImg.shape[1]], [terpImg.shape[0], terpImg.shape[1]]]
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
        #print(vh)
        print(A.dot(vh[8]))
        homographyMat = np.reshape(vh[8], (3, 3))
        
        self.transformImage(homographyMat, img, desiredCoordinates, cornerList)

    def findDistance(self,x1, y1, x2, y2):
        return math.pow(math.pow(x1-x2,2) + math.pow(y1-y2,2), 0.5)

    def transformImage(self, homographyMat, img, desiredCoordinates, cornerList):
        #cv2.imshow("frame1",img)
        homographyInv = inv(homographyMat)
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        ret,gray = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        width = self.findDistance(desiredCoordinates[1][0], desiredCoordinates[1][1], desiredCoordinates[0][0], desiredCoordinates[0][1])
        height =self.findDistance(desiredCoordinates[2][0], desiredCoordinates[2][1], desiredCoordinates[1][0], desiredCoordinates[1][1])
      
        newMat = np.ones([desiredCoordinates[1][0] - desiredCoordinates[0][0],desiredCoordinates[2][1] - desiredCoordinates[1][1],1],dtype=np.uint8)
        #newMat = np.zeros([int(height) + 200, int(width) + 200, 1],dtype=np.uint8)
        cur_x = 0
        cur_y = 0
        print(desiredCoordinates[1][1] - desiredCoordinates[2][1])
        for i in range(desiredCoordinates[1][1], desiredCoordinates[2][1]):
            cur_x=0
            for j in range(desiredCoordinates[0][0], desiredCoordinates[1][0]):
                curCord = np.array([[j],[i],[1]])
                newCord = homographyInv.dot(curCord)
                newCord[0] = newCord[0]/newCord[2]
                newCord[1] = newCord[1]/newCord[2]
                newMat[cur_x, cur_y] = gray[int(newCord[1])][int(newCord[0])]
                cur_x += 1 
            cur_y += 1
        plt.imshow(newMat, plt.cm.gray)   
        plt.show()  
       
    def cornerDetector(self):
        img = cv2.imread('frame15.jpg')
        terpImg = cv2.imread('testudo.png')
        img1 = img
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        ret,img = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)

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
  
        extTop = list(ext0)
        extBot = list(ext1)
        extLeft = list(ext2)
        extRight = list(ext3)
            
        cornerList = [ext0, ext3, ext2, ext1]
        desiredCoordinates = [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
        #desiredCoordinates = [(0, 0), (1, 0), (0, 1), (1, 1)]

        self.gethomographyMat(cornerList, desiredCoordinates, img1, terpImg)

        if cv2.waitKey(0) & 0xff == ord('q'):
            cv2.destroyAllWindows()
    
    def importImage(self):
        img = cv2.imread('frame15.jpg', 0)
        #ret,img = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
        cv2.imshow('s',img)
        dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(img))

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 200
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        dark_image_grey_fourier[mask_area] = 0

        dark_image_grey_fourier = np.fft.ifftshift(dark_image_grey_fourier)
        im_new = fftpack.ifft2(dark_image_grey_fourier)
        im_new = np.abs(im_new)
        
        plt.figure()
        plt.imshow(im_new, plt.cm.gray)
        plt.title('Reconstructed Image')
        plt.show()
        
p = AR_Detect()
p.cornerDetector()