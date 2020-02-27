import cv2.cv2 as cv2
import numpy as np


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
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (255, 0, 0), -1)
        cv2.imshow("circles",img)
    else:
        print("no clock detected !")



if __name__ == "__main__":
    img = cv2.imread("images/hard.jpg")
    img = resize(img,700)
    cv2.imshow("base",img)
    findCircles(img)



    cv2.waitKey()