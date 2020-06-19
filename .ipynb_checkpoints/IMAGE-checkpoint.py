import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage import color

class my_image:  
    @staticmethod
    def readimage(path):
        img_= cv2.imread(path)
        img_= cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        return img_
    @staticmethod
    def gauss_filter(img,ksize=7,sigma=100):
        kernel_vertically = cv2.getGaussianKernel(ksize, sigma)
        kernel_horizontaly = np.transpose(kernel_vertically)
        tmp = cv2.filter2D(img, -1, kernel_vertically)
        gsn = cv2.filter2D(tmp, -1, kernel_horizontaly)
        return gsn
    @staticmethod
    def increase_brightness(img, value=10):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    @staticmethod
    def edge_detection(img,mod='',blur_sigma=0.7):
        if mod=='sobel':
            #img_ = cv2.medianBlur(img.copy(), ksize=1)
            img_ =my_image.gauss_filter(img,7,blur_sigma)
            Gx=cv2.Sobel(img_.copy(),6,1,0)
            Gy=cv2.Sobel(img_.copy(),6,0,1)
            absx= cv2.convertScaleAbs(Gx)
            absy = cv2.convertScaleAbs(Gy)
            edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
            edge=my_image.increase_brightness(edge)
            edge_=color.rgb2gray(edge)
            return img_,edge_
    @staticmethod
    def make_gauss_noise(img,var=300,mean=0):
        row,col,ch= img.shape
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = img + gauss
        noisy[:][:][0] = np.clip(noisy[:][:][0], 0, 255)
        noisy[:][:][1] = np.clip(noisy[:][:][1], 0, 255)
        noisy[:][:][2] = np.clip(noisy[:][:][2], 0, 255)
        noisy=noisy.astype('uint8')
        return noisy
    
   
class Show:    
    @staticmethod
    def show_me(img,title='',mode=0,scale=1):
        if scale!=1:
            height=int(img.shape[0]*(scale))
            width=int(img.shape[1]*(scale))
            dimention=(width,height)
            img = cv2.resize(img, dimention, interpolation=cv2.INTER_AREA)
        if mode==0:
            plt.figure(figsize=(int(7*scale), int(14*scale)))
        if len(img.shape)==2:
            plt.imshow(img,'gray')    
        else:
            plt.imshow(img)
        plt.axis(False)
        plt.title(title)
        #plt.show()
    @staticmethod
    def compareim(img1,img2,title1='',title2='',size=1):
        plt.figure(figsize=(int(10*size), int(20*size)))
        plt.subplot(1,2,1)
        Show.show_me(img1,title1,1)
        plt.subplot(1,2,2)
        Show.show_me(img2,title2,1)