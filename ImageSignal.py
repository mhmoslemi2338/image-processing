from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import imutils
import cv2
from math import ceil
import matplotlib.pyplot as plt
from pyzbar import pyzbar



class ImageSignal:
    @staticmethod
    def segment_watershed(img,inv=True,distance=14,size=2,num=True):
        image=img.copy()
        rrr=ceil(min((image.shape)[1:2])/16 )
        shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        if inv==False:
            thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        else:
            thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh_=cv2.medianBlur(thresh,3)
        D = ndimage.distance_transform_edt(thresh_)
        localMax = peak_local_max(D, indices=False, min_distance=rrr,labels=thresh_)
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh_)
        for label in np.unique(labels):
            if label == 0: #background
                continue   #foreground
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            ((x, y), r) = cv2.minEnclosingCircle(c)
            [x_,y_,w_,h_]=cv2.boundingRect(c)
            #rec=np.int0(cv2.boxPoints(cv2.minAreaRect(c)))
            #draw number of segment
            if num==True:
                cv2.putText(image,"{}".format(label),(int(x)-10,int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 0, 0), size)
            #cv2.circle(tmp, (int(x), int(y)), int(r), (0, 255, 0), size_)  
            #cv2.drawContours(tmp,[rec],0,(0,0,255),1)
            cv2.rectangle(image,(x_-2,y_-2),(x_+w_+2,y_+h_+2), (0,0,255), size)
        return image



#########################################################################################################        
###############     from here functions are using for findng QR-code     ################################
#########################################################################################################        

    @staticmethod
    def four_point_transform(image, pts):
        (tl, tr, br, bl) = pts
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        maxWidth = max(int(widthA), int(widthB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped,maxWidth,maxHeight


    @staticmethod
    def QR_point(image,inv=True,ratio=20,num=False):
        #image=img.copy()
        min_=ceil(min((image.shape)[1:2])/ratio )
        shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        if inv==False:
            thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        else:
            thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=min_,labels=thresh)
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)
        pts=[]
        for label in np.unique(labels):
            if label != 0: #foreground
                mask = np.zeros(gray.shape, dtype="uint8")
                mask[labels == label] = 255
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                c = max(cnts, key=cv2.contourArea)
                ((x_, y_), r) = cv2.minEnclosingCircle(c)  #
                [x,y,w,h]=cv2.boundingRect(c)
                if num==True:
                    cv2.putText(image,"{}".format(label),(int(x_)-10,int(y_)),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 0, 0), 2)
                    P=np.array([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],dtype = "float32")
                else:
                    #P=np.array([(x-15,y+h+25),(x+w+15,y+h+25),(x+w+15,y-25),(x-15,y-25)],dtype = "float32")
                    P=np.array([(x,y+h),(x+w,y+h),(x+w,y),(x,y)],dtype = "float32")
                pts.append(P)
        return pts,image


    @staticmethod
    def QR_segment(img2):
        Rate=60
        Inv=True
        while(True):
            tmp=img2.copy()
            Next=True
            warped,Qr,size=[],[],[]
            pts,_= ImageSignal.QR_point(tmp,inv=Inv,ratio=Rate)
            for row in pts:
                [warp,width,height]=ImageSignal.four_point_transform(tmp,row)
                Qr.append(pyzbar.decode(warp))
                warped.append(warp)
                size.append(str(width)+'*'+str(height))
            for i in range(len(Qr)):
                if Qr[i]!=[]:
                    if min(int(width),int(height)) < min(np.shape(tmp)[0:2])-20:
                        Next=False
                        break
            if Next==False:
                break
            Rate=Rate-5
            if Rate==0:
                if Inv==False:
                    warped,Qr,size,i=[],[],[],-1
                    break
                else:
                    Inv=False
                    Rate=40

        return warped,Qr,size,i,pts



            