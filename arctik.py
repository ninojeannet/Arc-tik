import cv2.cv2 as cv2


# Resize the image at the given width keeping the width/height ratio
def resize(img,width):
    scale_ratio = width / img.shape[1]
    height = int(img.shape[0] * scale_ratio)
    dim = (width, height)
    img = cv2.resize(img,dim)
    return img

if __name__ == "__main__":
    img = cv2.imread("images/simple.jpg")
    img = resize(img,700)

    cv2.imshow("test",img)

    cv2.waitKey()