import cv2
import numpy as np

class Segmentation():
    '''分割提取轮廓的类及方法'''

    def extract_hsv(self,img):
        ''''method1：使用inRange方法，拼接红色mask0,mask1'''
        # global mask,img
        # img = cv2.imdecode(np.fromfile(pic, dtype=np.uint8), -1)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # rows, cols, channels = img.shape
        # 区间1
        lower_red = np.array([0, 43, 46])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
        # 区间2
        lower_red = np.array([156, 43, 46])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
        # 拼接两个区间
        mask = mask0 + mask1
        # 保存图片
        # cv2.imencode('.png', mask)[1].tofile(pic)

    def binaryzation(self,img):
        '''二值化阈值分割'''
        _,bin=cv2.threshold(img,150, 255,cv2.THRESH_BINARY)
        return bin
    def adaptive_binaryzation(self):
        '''自适应二值化阈值分割'''
        pass
    # 分割去除红色
    def remove_red_seal(self, image):
        """
        去除红色印章
        """

        # 获得红色通道
        blue_c, green_c, red_c = cv2.split(image)
        # cv2.imshow('red',red_c)
        # 多传入一个参数cv2.THRESH_OTSU，并且把阈值thresh设为0，算法会找到最优阈值
        thresh, ret = cv2.threshold(red_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 实测调整为95%效果好一些
        filter_condition = int(thresh * 0.95)

        _, red_thresh = cv2.threshold(red_c, filter_condition, 255, cv2.THRESH_BINARY)

        # 把图片转回 3 通道
        # result_img = np.expand_dims(red_thresh, axis=2)
        # result_img = np.concatenate((result_img, result_img, result_img), axis=-1)

        return red_thresh  # result_img

class ImgSourceInit():
    def img_init(self,filename):
        img = cv2.imread(filename)
        return img

    def video_init(self,camid):
        global cam
        cam = cv2.VideoCapture(camid)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# if __name__ == '__main__':
#     Img = ImgSourceInit()
#     Img.video_init(1)
#     hsvs = Segmentation()
#     while(1):
#         ret,img = cam.read()
#         # img=hsvs.binaryzation(img)
#         img = hsvs.remove_red_seal(img)
#         # cv2.imwrite("D:/test/result.png",rm_img)
#         cv2.imshow("winName",img)
#         if cv2.waitKey(1)==27:
#             break
#     cam.release()
#     cv2.destroyAllWindows()