from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import imutils
import cv2
from math import ceil



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

        
