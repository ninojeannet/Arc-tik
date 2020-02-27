import cv2.cv2 as cv2


if __name__ == "__main__":
    img = cv2.imread("images/medium.png")
    
    width = 700
    scale_ratio = width / img.shape[1]
    height = int(img.shape[0] * scale_ratio)
    dim = (width, height)
    img = cv2.resize(img,dim)
    cv2.imshow("test",img)
    cv2.waitKey()