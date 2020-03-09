import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt


# Resize the image at the given width keeping the width/height ratio
def resize(img,width):
    scale_ratio = width / img.shape[1]
    height = int(img.shape[0] * scale_ratio)
    dim = (width, height)
    img = cv2.resize(img,dim)
    return img

def findCircles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1.2,100)

    if circles is not None:
        radius = 0
        center = (0,0)
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (255, 0, 0), -1)
            radius = r
            center = (x,y)
        return img,center,radius
    else:
        print("no clock detected !")


def createPolar(img,center,radius):
    imgSize = (radius,360)
    polarImg = cv2.warpPolar(img,imgSize,center,radius,flags=0)
    return polarImg

def findClockHand(img):
    # resize 64 - 171
    croppedImage = img[0:360,64:171]
    # Convert to grayscale
    grayscale = cv2.cvtColor(croppedImage,cv2.COLOR_RGB2GRAY)
    # threshold ~25
    retval, thresh = cv2.threshold(grayscale,15,255,cv2.THRESH_BINARY)
    # isolate each clock hand using length ?
    return makeHisto(thresh)
    #test = imgPolar[9:10]
    #cv2.imshow("gray",thresh)

def makeHisto(img):
    nbLine,nbColumn  = img.shape
    histo = dict()
    for i in range(nbLine):
        histo[i] = 0

    for i in range(nbLine):
        for j in range(nbColumn):
            if img[i,j] == 0:
                histo[i] = histo[i]+1

    mean = int(findMean(histo,0,nbLine)[0])
    histo2 = dict(list(histo.items())[mean:])
    histo1 = dict(list(histo.items())[:mean])
    #plot_histogram_from_dict(histo1)
    #plot_histogram_from_dict(histo2)
    
    mean1,length1 = findMean(histo1,0,mean)
    mean2,length2 = findMean(histo2,mean,nbLine)
    hourMean = mean1
    minuteMean= mean2
    print(length1)
    print(length2)
    if length1 > length2:
        minuteMean = mean1
        hourMean = mean2
    else:
        minuteMean = mean2
        hourMean = mean1
    return hourMean,minuteMean


def plot_histogram_from_dict(dict):
    plt.bar(dict.keys(), dict.values(), color='g')
    plt.show()

def findMean(histo,start,nbLine):
    sumHist = 0 

    for i in range(start,nbLine):
        sumHist = sumHist + (histo[i]*i)
    mean = sumHist / sum(histo.values())
    
    length = findLength(histo,mean)
    #print(mean)
    return mean,length

def findLength(histo,mean):
    mean = int(mean)
    sum=0
    print("mean", mean)
    print("taille de l'histo ",len(histo))
    print("début de l'index ", list(histo.keys())[0])
    start = list(histo.keys())[0]
    stop = list(histo.keys())[0] + len(histo)
    print("histogramme ",histo)

    for i in range(mean-1,mean+2):
        #print("i ",i)
        if i >=start and i < stop: 
            sum+=histo[i]
        #print("s ",histo[i%len(histo) + list(histo.keys())[0]])
        
    length = sum / 3
    print("----------------------------------")

    return max(histo)

if __name__ == "__main__":
    img = cv2.imread("images/medium.png")
    img = resize(img,700)
    #cv2.imshow("base",img)

    img,center,radius = findCircles(img)
    #cv2.imshow("circles",img)

    imgPolar = createPolar(img,center,radius)
    cv2.imshow("polar",imgPolar)
    cv2.imwrite("polar.jpg",imgPolar)
    clockHandHour, clockHandMinute = findClockHand(imgPolar)

    minute =  int((clockHandMinute / 360 * 60 + 15) % 60)
    hour= int((clockHandHour / 360 * 12 + 3) % 12)

    print("Il est ",hour,"h",minute)

    cv2.waitKey()