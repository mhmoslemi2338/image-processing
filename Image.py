import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage import color

class segmentation:
    @staticmethod    
    def ColorSegmentator(image, min_color, max_color):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        color_low=(min_color,40,40)
        color_high=(max_color,255,253) 
        mask = cv2.inRange(hsv_image, color_low, color_high)
        result = cv2.bitwise_and(image, image, mask=mask)
        return result           
   
    @staticmethod
    def LinesDetector(img,minlenght=0,r=255,g=0,b=0):
        tmp=img.copy()
        img_=color.rgb2gray(img.copy())
        Gx=cv2.Sobel(img_.copy(),6,1,0,ksize=7)
        Gy=cv2.Sobel(img_.copy(),6,0,1,ksize=7)
        absx= cv2.convertScaleAbs(Gx)
        absy = cv2.convertScaleAbs(Gy)
        edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
        th1 = cv2.threshold(edge,19,150,cv2.THRESH_BINARY)
        lines=cv2.HoughLinesP(th1[1],1,np.pi/(250),80,30,10)
        for i in range(int(len(lines))):
            [x1,y1,x2,y2]=lines[i][0]
            w=((x1-x2)**2+(y1-y2)**2)
            if w >= minlenght**2:
                cv2.line(tmp,(x1,y1), (x2,y2),(r,g,b) ,2)
        return tmp
    @staticmethod
    def PolygonDetector(Img,maxside=12,r=255,g=0,b=0,thickness=5):
        img2=Img.copy()
        img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img=cv2.blur(img,(3,3))
        img2=cv2.blur(img2,(3,3))
        _,threshold = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY) 
        contours,_=cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        for cnt in contours : 
            area = cv2.contourArea(cnt) 
            if area > 200: 
                approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 
                if(len(approx) >2 and len(approx) <maxside+1 ): 
                    cv2.drawContours(img2, [approx], 0, (b, g, r), thickness) 
        return img2
    
    @staticmethod
    def camera_color():
        cap = cv2.VideoCapture(0)
        while(True):
            _, frame = cap.read()
            frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result_orange=segmentation.ColorSegmentator(frame1,0,15)
            result_red=segmentation.ColorSegmentator(frame1,119,179)
            result_blue=segmentation.ColorSegmentator(frame1,85,120)
            result_yellow=segmentation.ColorSegmentator(frame1,22,35)
            result_green=segmentation.ColorSegmentator(frame1,36,80)

            result_red=cv2.cvtColor(result_red, cv2.COLOR_RGB2BGR)
            result_orange=cv2.cvtColor(result_orange, cv2.COLOR_RGB2BGR)
            result_blue=cv2.cvtColor(result_blue, cv2.COLOR_RGB2BGR)
            result_yellow=cv2.cvtColor(result_yellow, cv2.COLOR_RGB2BGR)
            result_green=cv2.cvtColor(result_green, cv2.COLOR_RGB2BGR)


            frame = cv2.resize(frame, (365, 365)) 
            result_red = cv2.resize(result_red, (365, 365)) 
            result_orange = cv2.resize(result_orange, (365, 365)) 
            result_blue = cv2.resize(result_blue, (365, 365)) 
            result_yellow = cv2.resize(result_yellow, (365, 365)) 
            result_green = cv2.resize(result_green, (365, 365)) 

            cv2.imshow('frame',frame)
            cv2.imshow('red & pink',result_red)
            cv2.imshow('orange',result_orange)
            cv2.imshow('blue',result_blue)
            cv2.imshow('yellow',result_yellow)
            cv2.imshow('green',result_green)
            k = cv2.waitKey(5) & 0xFF # Escape key
            if k == 27:
                break
        cap = cv2.VideoCapture(1)
        cv2.destroyAllWindows()

    @staticmethod
    def camera_Polygon(maxside=20):
        cap = cv2.VideoCapture(0)
        while(True):
            _, frame = cap.read()
            #frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            s=segmentation.PolygonDetector(frame,maxside,r=255,g=0,b=0,size=2)
            #frame = cv2.resize(frame, (365, 365))
            #s = cv2.resize(s, (700, 700))
            cv2.imshow('frame',frame)
            cv2.imshow('Polygon Detection with maxSide %d' %maxside,s)
            k = cv2.waitKey(5) & 0xFF # Escape key
            if k == 27:
                break
        cap = cv2.VideoCapture(1)
        cv2.destroyAllWindows()

    @staticmethod
    def camera_Line(minlenght=20):
        cap = cv2.VideoCapture(0)
        while(True):
            _, frame = cap.read()
            frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            s=segmentation.LinesDetector(frame,minlenght)

            cv2.imshow('frame',frame)
            cv2.imshow('line detection with minLenrh %d' %minlenght,s)
            k = cv2.waitKey(5) & 0xFF # Escape key
            if k == 27:
                break
        cap = cv2.VideoCapture(1)
        cv2.destroyAllWindows()




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
        lines=cv2.HoughLinesP(th1,1,np.pi/(720),80,30,10)
        line_detect=cv2.cvtColor(th1,cv2.COLOR_GRAY2BGR )
        for i in range(int(len(lines))):
            for x1,y1,x2,y2 in lines[i]: 
                cv2.line(line_detect,(x1,y1), (x2,y2), (255,0,0),2)
        return img,edge,line_detect
    @staticmethod
    def line_detection2(img,r=255,g=0,b=0,thickness=2):
        tmp=img.copy()
        _,edge=my_image.edge_detection(tmp.copy(),'sobel',blur_sigma=0.7,floor=20,thresh_=0)
        _,th1 = cv2.threshold(edge,19,150,cv2.THRESH_BINARY)
        lines=cv2.HoughLinesP(th1,1,np.pi/(1000),80,30,10)
        for i in range(int(len(lines))):
            for x1,y1,x2,y2 in lines[i]: 
                cv2.line(tmp,(x1,y1), (x2,y2), (r,g,b),thickness)
        return tmp
    
    @staticmethod
    def circle_detection(origin,k1=3,k2=5,k3=11,thickness=3):
        img=origin.copy()
        gray=cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,k1)
        edge=cv2.Canny(gray,30,255)
        edge=cv2.GaussianBlur(edge,(k2, k2), 0.4);
        edge1 = cv2.blur(edge, (k3, k3))

        gray_blurred=4*edge1
        tmp=20
        while(True):
            detected_circles = cv2.HoughCircles(gray_blurred,  cv2.HOUGH_GRADIENT, 1, tmp,
                                             param1 = 20, param2 = 100, minRadius = 0, maxRadius = 0) 
            if len(detected_circles[0, :])>8:
                tmp+=20
            else:
                break
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], int(pt[2] )
            cv2.circle(img, (a, b), r, (0, 255, 0), thickness) 
            cv2.circle(img, (a, b), 1, (0, 0, 255), thickness) 
        return img

'''
        gray=cv2.cvtColor(origin.copy(),cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,3)
        edge=cv2.Canny(gray,50,255)
        edge=cv2.GaussianBlur(edge,(11, 11), 0.4);
        
        #trash,th1 = cv2.threshold(edge,10,200,cv2.THRESH_BINARY)
        tmp=20
        while(True):
            circles = cv2.HoughCircles(edge,cv2.HOUGH_GRADIENT,1,20,
                                       param1=20,param2=tmp,minRadius=0,maxRadius=0)
            if circles.shape[1]>20:
                tmp+=10
            else:
                break  
        circles = np.uint16(np.around(circles))
        cimg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        for i in circles[0,:]:
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(cimg,(i[0],i[1]),2,(255,0,0),2)
        return cimg
'''
    
    
class Show:    
    @staticmethod
    def show_me(img,title='',mode='',scale=1):
        if scale!=1:
            height=int(img.shape[0]*(scale))
            width=int(img.shape[1]*(scale))
            dimention=(width,height)
            img = cv2.resize(img, dimention, interpolation=cv2.INTER_AREA)
        if mode=='':
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
            Show.show_me(img1,title1,mode='compare')
            plt.subplot(1,3,2)
            Show.show_me(img2,title2,mode='compare')
            plt.subplot(1,3,3)
            Show.show_me(img3,title3,mode='compare')
        else:
            plt.figure(figsize=(int(10*size), int(20*size)))
            plt.subplot(1,2,1)
            Show.show_me(img1,title1,1)
            plt.subplot(1,2,2)
            Show.show_me(img2,title2,1)
    
    @staticmethod
    def show_segments(warped,size,Qr):
        for i in range(0,len(warped),6):
            tmp=[]
            size_=[]
            if Qr==0:
                Qr=[0 for i in range(len(warped))]
            qr=[]
            for j in range(6):
                try:
                    tmp.append(warped[i+j])
                    size_.append(size[i+j])
                    if Qr[i+j]==[]:
                        qr.append(False)
                    elif Qr[i+j]==0:
                        qr.append('')
                    else:
                        qr.append(True)
                except:
                    break
            l=len(tmp)
            for j in range(l):
                pl=100+10*l+j+1
                plt.subplot(pl)
                plt.imshow(cv2.cvtColor(tmp[j],cv2.COLOR_BGR2RGB));plt.axis(False);
                if qr[j]!='':
                    plt.title(size_[j]+'\n'+'Qr:'+str(qr[j]));
                else:
                    plt.title(size_[j]);
            plt.show()