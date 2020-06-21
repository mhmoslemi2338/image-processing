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
    def threshhold(img,floor=26,limit=200):
        tmp=img.copy()
        for i in range(int(len(tmp))):
            if max(tmp[i])<26:
                continue
            for j in range(int(len(tmp[0]))):
                if tmp[i][j]>=floor and tmp[i][j]<=limit:
                    tmp[i][j]=tmp[i][j]+50
        return tmp
    @staticmethod
    def edge_detection(img,mod='',blur_sigma=0.7,floor=20,thresh_=1):
        if mod=='sobel':
            img_=color.rgb2gray(img.copy())
            img_ = cv2.medianBlur(img_, ksize=1)
            img_ =my_image.gauss_filter(img_,7,blur_sigma)
            Gx=cv2.Sobel(img_.copy(),6,1,0,ksize=7)
            Gy=cv2.Sobel(img_.copy(),6,0,1,ksize=7)
            absx= cv2.convertScaleAbs(Gx)
            absy = cv2.convertScaleAbs(Gy)
            edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
            # edge_=color.rgb2gray(edge)
            #ret2,edge = cv2.threshold(edge,70,150,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            if thresh_==1:
                edge=my_image.threshhold(edge)
            #ret1,edge2 = cv2.threshold(edge,20,255,cv2.THRESH_BINARY)
            return img_,edge
        if mod=='laplace':
            img_ = cv2.medianBlur(img.copy(), ksize=5)
            img_ =my_image.gauss_filter(img_,7,blur_sigma)
            #img_ = cv2.GaussianBlur(img.copy(), (3, 3), blur_sigma)
            img_gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            dst = cv2.Laplacian(img_gray, ddepth=6, ksize=5)
            edge = cv2.convertScaleAbs(dst)
            edge = cv2.medianBlur(edge, ksize=3)
           # edge=my_image.threshhold(abs_dst,floor)
            return img_,edge
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
    @staticmethod
    def line_detection(img):
        trash,edge=my_image.edge_detection(img,'sobel',blur_sigma=0.7,floor=20,thresh_=0)
        trash,th1 = cv2.threshold(edge,19,150,cv2.THRESH_BINARY)
        lines=cv2.HoughLinesP(th1,1,np.pi/(1800),80,30,10)
        line_detect=cv2.cvtColor(th1,cv2.COLOR_GRAY2BGR )
        for i in range(int(len(lines))):
            for x1,y1,x2,y2 in lines[i]: 
                cv2.line(line_detect,(x1,y1), (x2,y2), (255,0,0),2)
        return img,edge,line_detect

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
    def compareim(img1,img2,title1='',title2='',size=1,triple=0,img3=[],title3=''):
        if triple!=0:
            plt.figure(figsize=(int(10*size), int(20*size)))
            plt.subplot(1,3,1)
            Show.show_me(img1,title1,1)
            plt.subplot(1,3,2)
            Show.show_me(img2,title2,1)
            plt.subplot(1,3,3)
            Show.show_me(img3,title3,1)
        else:
            plt.figure(figsize=(int(10*size), int(20*size)))
            plt.subplot(1,2,1)
            Show.show_me(img1,title1,1)
            plt.subplot(1,2,2)
            Show.show_me(img2,title2,1)
