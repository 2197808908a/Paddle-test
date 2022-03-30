import cv2 as cv
import numpy

brightness_threshold = 200

def channel_subtract(image):
    blue_c, green_c, red_c = cv.split(image)
    red_sub_blue = red_c - blue_c
    blue_sub_red = blue_c -red_c
    # cv.imshow("red_c",blue_c)
    red_sub_blue = cv.GaussianBlur(red_sub_blue,(3,3),1.5)
    blue_sub_red = cv.GaussianBlur(blue_sub_red,(3,3),1.5)
    normalize_red = cv.normalize(red_sub_blue,0,255,cv.NORM_MINMAX)
    normalize_blue = cv.normalize(blue_sub_red,0,255,cv.NORM_MINMAX)
    _,red_thresh = cv.threshold(normalize_red,brightness_threshold,255,cv.THRESH_OTSU)
    _,blue_thresh = cv.threshold(blue_sub_red,brightness_threshold,255,cv.THRESH_OTSU)
    cv.imshow("red_sub_img", red_thresh)

def numpy_channel_subtract(image):
     # 获取 B 通道
    bImg = image.copy()  # 获取 BGR
    bImg[:, :, 1] = 0  # G=0
    bImg[:, :, 2] = 0  # R=0
    
       # 获取 G 通道
    gImg = image.copy()  # 获取 BGR
    gImg[:, :, 0] = 0  # B=0
    gImg[:, :, 2] = 0  # R=0
    
    # 获取 R 通道
    rImg = image.copy()  # 获取 BGR
    rImg[:, :, 0] = 0  # B=0
    rImg[:, :, 1] = 0  # G=0
    # cv.imshow("numpy",bImg)
    red_sub_blue = rImg - bImg
    blue_sub_red = bImg -rImg
    # cv.imshow("numpy1",red_sub_blue)
    # red_sub_blue = cv.GaussianBlur(red_sub_blue,(3,3),1.5)
    # blue_sub_red = cv.GaussianBlur(blue_sub_red,(3,3),1.5)
    # normalize_red = cv.normalize(red_sub_blue,0,255,cv.NORM_MINMAX)
    # normalize_blue = cv.normalize(blue_sub_red,0,255,cv.NORM_MINMAX)
    # _,red_thresh = cv.threshold(normalize_red,brightness_threshold,255,cv.THRESH_OTSU)
    # _,blue_thresh = cv.threshold(blue_sub_red,brightness_threshold,255,cv.THRESH_OTSU)
    # cv.imshow("red_sub_img", red_thresh)
def test(img1):
    # img1 = cv.imread("1.jpg")
    B,G,R = cv.split(img1)
    cv.imshow("Red", R)
    cv.imshow("Green", G)
    cv.imshow("Blue",B)
    red = R - B
    cv.imshow("red",red)
    red1 = cv.subtract(R,B)
    cv.imshow("red1",red1)
    blue = cv.subtract(B,R)
    cv.imshow("blue",blue)
    # print("R_shape:",G.shape,"  G_shape:",G.shape,"  B_shape:",B.shape)


if __name__ == '__main__':
    # cap=cv.VideoCapture('armor.mp4')
    while(1):
        img = cv.imread("1.png")
        # ret,img = cap.read()
        # channel_subtract(img)
        # numpy_channel_subtract(img)
        test(img)
        # dst_dilate, frame= #read_morphology(cap)
        # find_contours(dst_dilate, frame)
        # cv.imshow("winName", dst_dilate)
        # cv.imshow("winName1", frame)
        cv.imshow("img",img)
        if cv.waitKey(1) == 27:
            break