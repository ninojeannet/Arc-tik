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

def isolateClockHand(img):
    # resize 64 - 171
    croppedImage = img[0:360,64:171]
    # Convert to grayscale
    grayscale = cv2.cvtColor(croppedImage,cv2.COLOR_RGB2GRAY)
    # threshold ~25
    retval, thresh = cv2.threshold(grayscale,15,255,cv2.THRESH_BINARY)
    # isolate each clock hand using length ?
    makeHisto(thresh)
    #test = imgPolar[9:10]
    cv2.imshow("gray",thresh)

def makeHisto(img):
    nbLine,nbColumn  = img.shape
    histo = dict()
    for i in range(nbLine):
        histo[i] = 0

    for i in range(nbLine):
        for j in range(nbColumn):
            if img[i,j] == 0:
                histo[i] = histo[i]+1

    mean = int(findMean(histo,0,nbLine))
    histo2 = dict(list(histo.items())[mean:])
    histo1 = dict(list(histo.items())[:mean])
    #plot_histogram_from_dict(histo1)
    #plot_histogram_from_dict(histo2)
    
    mean1 = findMean(histo1,0,mean)
    mean2 = findMean(histo2,mean,nbLine)

    

def plot_histogram_from_dict(dict):
    plt.bar(dict.keys(), dict.values(), color='g')
    plt.show()

def findMean(histo,start,nbLine):
    sumHist = 0 

    for i in range(start,nbLine):
        sumHist = sumHist + (histo[i]*i)
    mean = sumHist / sum(histo.values())
    print(mean)
    return mean

if __name__ == "__main__":
    img = cv2.imread("images/mondaine.jpg")
    img = resize(img,700)
    #cv2.imshow("base",img)

    img,center,radius = findCircles(img)
    #cv2.imshow("circles",img)

    imgPolar = createPolar(img,center,radius)
    cv2.imshow("polar",imgPolar)
    cv2.imwrite("polar.jpg",imgPolar)
    isolateClockHand(imgPolar)
    cv2.waitKey()