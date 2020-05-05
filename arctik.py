'''
    Projet Arc'tik
    Nino Jeannet & Sébastien Peiris
    Traitement d'images
    HE-Arc
    05.05.2020
'''

import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt


# Resize the image at the given width keeping the width/height ratio
def resize(img,width):
    scale_ratio = width / img.shape[1]
    height = int(img.shape[0] * scale_ratio)
    dim = (width, height)
    img = cv2.resize(img,dim)
    return img,scale_ratio

# Find the circular shapes in an image
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
        #cv2.imshow("e",img)
        return img,center,radius
    else:
        #print("no clock detected !")
        return None

# Refocus the image on the clock
def centerClock(img,center,radius):
    margin = 0
    img = img[center[1]-radius-margin:center[1]+radius+margin,center[0]-radius-margin:center[0]+radius+margin]
    img,scale_ratio = resize(img,700)
    radius = int(radius*scale_ratio)
    center = (radius,radius)
    #cv2.imshow("re",img)

    return img,center,radius

# Apply a polar transformation to the clock
def createPolar(img,center,radius):
    imgSize = (radius,360)
    polarImg = cv2.warpPolar(img,imgSize,center,radius,flags=0)
    #cv2.imshow("res",polarImg)
    return polarImg

# Applies a grayscale filter on the clock. Returns an histogram with the clock's 2 hands
def findClockHand(img,radius,center):
    #ratio = int(radius / 100 * 60)
    # resize 64 - 171
    croppedImage = img[0:360,90:220]
    #cv2.imshow("cropped",croppedImage)
    # Convert to grayscale
    grayscale = cv2.cvtColor(croppedImage,cv2.COLOR_RGB2GRAY)
    # threshold ~25
    retval, thresh = cv2.threshold(grayscale,15,255,cv2.THRESH_BINARY)
    # isolate each clock hand using length ?
    #cv2.imshow("gray",thresh)
    return makeHisto(thresh)
    #test = imgPolar[9:10]
    

#Returns the hours hand and the minutes hand positions
def makeHisto(img):
    nbLine,nbColumn  = img.shape
    histo = dict()
    for i in range(nbLine):
        histo[i] = 0

    for i in range(nbLine):
        for j in range(nbColumn):
            if img[i,j] == 0:
                histo[i] = histo[i]+1

    # Separation by the mean of the histogram
    mean = int(findMean(histo,0,nbLine)[0])
    histoBottom = dict(list(histo.items())[mean:])
    histoTop = dict(list(histo.items())[:mean])

    meanTop,length1 = findMean(histoTop,0,mean)
    meanBottom,length2 = findMean(histoBottom,mean,nbLine)

    # Double separation in case on hand is on the 3 mark 
    histoQ1 = dict(list(histoTop.items())[:meanTop])
    histoQ2 = dict(list(histoTop.items())[meanTop:])
    histoQ3 = dict(list(histoBottom.items())[:meanBottom-mean])
    histoQ4 = dict(list(histoBottom.items())[meanBottom-mean:])
    #plot_histogram_from_dict(histoQ4)
    meanQ1,lengthQ1 = findMean(histoQ1,0,meanTop)
    meanQ2,lengthQ2 = findMean(histoQ2,meanTop,mean)
    meanQ3,lengthQ3 = findMean(histoQ3,mean,meanBottom)
    meanQ4,lengthQ4 = findMean(histoQ4,meanBottom,nbLine-1)

    #print("Q1 mean: ",meanQ1," length: ",lengthQ1)
    #print("Q2 mean: ",meanQ2," length: ",lengthQ2)
    #print("Q3 mean: ",meanQ3," length: ",lengthQ3)
    #print("Q4 mean: ",meanQ4," length: ",lengthQ4)

    histostats =dict()
    histostats[lengthQ1] = meanQ1
    histostats[lengthQ2] = meanQ2
    histostats[lengthQ3] = meanQ3
    histostats[lengthQ4] = meanQ4

    meanLength = int((lengthQ1+lengthQ2+lengthQ3+lengthQ4)/4)

    # hourMean and minuteMean give us the positions of both hands
    hourMean = 0
    minuteMean = 0
    for length in histostats.items():
        if length[0] > meanLength:
            minuteMean = length[1]
        else:
            hourMean = length[1]


    '''
        if length1 > length2:
            minuteMean = meanTop
            hourMean = meanBottom
        else:
            minuteMean = meanBottom
            hourMean = meanTop
    '''
    return hourMean,minuteMean

# Method that plots a histogram from a given dictionnary
def plot_histogram_from_dict(dict):
    plt.bar(dict.keys(), dict.values(), color='g')
    plt.show()

# Finds the mean of a histogram
def findMean(histo,start,nbLine):
    sumHist = 0 
    for i in range(start,nbLine):
        sumHist = sumHist + (histo[i]*i)
    mean = sumHist / sum(histo.values())
    
    length = findLength(histo,mean)
    return int(mean),length

# Finds the length of a histogram
def findLength(histo,mean):
    mean = int(mean)
    sum=0
    #print("mean", mean)
    #print("taille de l'histo ",len(histo))
    #print("début de l'index ", list(histo.keys())[0])
    start = list(histo.keys())[0]
    stop = list(histo.keys())[0] + len(histo)
    #print("histogramme ",histo)

    for i in range(mean-1,mean+2):
        #print("i ",i)
        if i >=start and i < stop: 
            #print(histo[i])
            sum+=histo[i]
        #print("s ",histo[i%len(histo) + list(histo.keys())[0]])
        
    length = sum / 3
    return max(list(histo.values()))

# Processes every images in the "images/" repository
def tryEachClock():
    import os
    for filename in os.listdir('images/'):
        print("heure de l'horloge",filename,":")
        processImage(filename)

# Processes a given images and returns the right time if a clock is found
def processImage(filename):
    img = cv2.imread("images/"+filename)
    img,scale_ratio = resize(img,700)
    #cv2.imshow("base",img)
    
    try:
        img,center,radius = findCircles(img)
        #cv2.imshow("circles",img)
        img,center,radius = centerClock(img,center,radius)
        imgPolar = createPolar(img,center,radius)

        #cv2.imshow("polar",imgPolar)
        cv2.imwrite("polar.jpg",imgPolar)
        clockHandHour, clockHandMinute = findClockHand(imgPolar,radius,center)

        minute =  int((clockHandMinute / 360 * 60 + 15) % 60)
        hour= int((clockHandHour / 360 * 12 + 3) % 12)

        print("Il est ",hour,"h",minute)
    except:
        print("Pas d'horloge détectée")

if __name__ == "__main__":
    tryEachClock()
    #processImage("medium.png")

    cv2.waitKey()