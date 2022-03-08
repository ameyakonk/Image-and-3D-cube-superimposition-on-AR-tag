from unittest import skip
import numpy as np
from numpy import linalg as LA
from scipy import fftpack
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image as im
from numpy.linalg import inv
import os
from PIL import Image

class AR_Detect:

    def imageFFT(self):

        img = cv2.imread('frame15.jpg', 0)
        ret,img = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
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
        plt.imshow(im_new, plt.cm.gray)
        plt.show()
        
p = AR_Detect()
p.imageFFT()