from re import X
from unittest import skip
import numpy as np
from numpy import linalg as LA
from scipy import fftpack
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image as im
from numpy.linalg import inv
from numpy import asarray

class AR_Detect:
    def __init__(self):
         self.terpImg = cv2.imread('testudo.png')
         self.repeat_count = 0
         self.skipFrame = False
               
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

    def gethomographyMat(self, cornerList, desiredCoordinates,image_height, image_width, img):
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
        self.transformImage(homographyMat, img, desiredCoordinates,image_height, image_width)

    def findDistance(self,x1, y1, x2, y2):
        return math.pow(math.pow(x1-x2,2) + math.pow(y1-y2,2), 0.5)
    
    def transformImage(self, homographyMat, img, desiredCoordinates, image_height, image_width):
        homographyInv = inv(homographyMat)
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        ret,gray = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        
        #newMat = np.ones([desiredCoordinates[1][0] - desiredCoordinates[0][0],desiredCoordinates[2][1] - desiredCoordinates[1][1],1],dtype=np.uint8)
        cur_x = 0
        cur_y = 0
        dict_ = {}
        
        (y, x) = np.mgrid[desiredCoordinates[1][1]:desiredCoordinates[2][1],desiredCoordinates[0][0]:desiredCoordinates[1][0]]
        print(y[0])
        print(x[0])
        one = np.ones(y.shape[:2][0]*x.shape[:2][1])
        arr = np.stack((np.ravel(x), np.ravel(y), np.ravel(one)), axis=-1)
        print(arr)
        newArr = arr.dot(homographyInv.T)
        X = newArr[:,0]/newArr[:,2]
        Y = newArr[:,1]/newArr[:,2]
        print(X[0])
        print(Y[0])
        
        X_min = np.amin(X)
        X_max = np.amax(X)
        
        Y_min = np.amin(Y)
        Y_max = np.amax(Y)
        newImg = asarray(gray)
        newImg = np.resize(newImg,(1080, 1920, 1))
        newMat = np.ones([int(Y_max-Y_min)+1,int(X_max - X_min)+1,1],dtype=np.uint8)
        
        # print(newMat.shape[:2])
         
        newMat[0:202, 0:206] = newImg[int(Y_min):int(Y_max), int(X_min):int(X_max)]
    
        #newImg[int(Y_min):int(Y_max), int(X_min):int(X_max)] = [255, 0, 0]
        
        plt.imshow(newMat, plt.cm.gray)   
        plt.show() 
        # print(Y)

        # for i in range(desiredCoordinates[1][1], desiredCoordinates[2][1]):
        # #     cur_x=0
        #      for j in range(desiredCoordinates[0][0], desiredCoordinates[1][0]):
        #          newCord = 
        #          if 0 <= int(newCord[1]) < 1080 and 0 <= int(newCord[0]) < 1920:
        #              newMat[cur_x, cur_y] = gray[int(newCord[1])][int(newCord[0])]
        #             dict_[(cur_x, cur_y)] = curCord
        #             cur_x += 1 
        #     cur_y += 1
        # self.repeat_count += 1

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
        self.repeat_count += 1
        plt.imshow(newMat, plt.cm.gray)   
        plt.show()  
        print(self.repeat_count)
        if(self.repeat_count < 2):
            self.terpImg, skipFrame = self.detectArucoOrientation(newMat)
            cv2.imshow("frame", self.terpImg)
            if cv2.waitKey(1) & 0xff == ord('q'):
                cv2.destroyAllWindows()
        #if not skipFrame:
 
        for i in range(int(self.terpImg.shape[:2][1])):
            for j in range(int(self.terpImg.shape[:2][0])):
                 if (j,i) in dict_:
                    x = dict_[(j, i)][0]
                    y = dict_[(j, i)][1]
                    curCord = np.array([[x],[y],[1]], dtype=object)
                    newCord = np.array([[0], [0], [1]])
                    newCord = homographyInv.dot(curCord)
                    newCord = np.matmul(homographyInv, curCord)
                    newCord[0] = newCord[0]/newCord[2]
                    newCord[1] = newCord[1]/newCord[2]
                    if 0 <= int(newCord[1]) < 1080 and 0 <= int(newCord[0]) < 1920:
                        img[int(newCord[1])][int(newCord[0])] = self.terpImg[j, i]

        dict_.clear()
        cv2.imshow("frame", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            cv2.destroyAllWindows()

    def detectArucoOrientation(self, newMat):
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
            # print(binary[0])
            # print(binary[1])
            # print(binary[2])
            # print(binary[3])  
            arucoId = binary[0]*math.pow(2,0) + binary[1]*math.pow(2,1) + binary[2]*math.pow(2,2) + binary[3]*math.pow(2,3)
            print(arucoId)
            if arucoId == 13:
                break
            if valid_counter >= 4:
                skipFrame = True
                return self.terpImg, skipFrame
            newMat = cv2.rotate(newMat, cv2.ROTATE_90_CLOCKWISE)
            self.terpImg = cv2.rotate(self.terpImg, cv2.ROTATE_90_CLOCKWISE)
            binary.clear()
            # break
        self.terpImg = cv2.resize(self.terpImg, newMat.shape[:2], interpolation = cv2.INTER_AREA)
        return self.terpImg, skipFrame
        
        # print("exited")
        # plt.imshow(newMat, plt.cm.gray)   
        # plt.show()  
      
    def cornerTest(self):
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
        
                img1 = cv2.circle(img1, ext0, 5, (255,0, 0), -1)
                img1 = cv2.circle(img1, ext1, 5, (255,0, 0), -1)
                img1 = cv2.circle(img1, ext2, 5, (255,0, 0), -1)
                img1 = cv2.circle(img1, ext3, 5, (255,0, 0), -1)
            #     cv2.imshow("frame", img1)
            # if cv2.waitKey(1) & 0xff == ord('q'):
            #     cv2.destroyAllWindows()
        
    def cornerDetector(self):
        cap = cv2.VideoCapture('1tagvideo.mp4')
   
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video  file")
        result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (1920, 1080))
        # Read until video is completed
        while(cap.isOpened()):
            ret, img = cap.read()
            if ret == True:
                #img = cv2.imread('frame15.jpg')
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
                
                #cv2.imshow("frame", img1)
                cornerList = [ext0, ext3, ext2, ext1]
                print(cornerList)
                image_width, image_height =  np.abs(x_min - x_max), np.abs(y_min -y_max)
                desiredCoordinates = [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
                #desiredCoordinates = [(0, 0), (0, image_width), (image_height, 0), (image_width, image_height)]
                self.gethomographyMat(cornerList, desiredCoordinates,image_height, image_width, img1)

            # if cv2.waitKey(1) & 0xff == ord('q'):
            #     cv2.destroyAllWindows()
    
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