# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import cv2 as cv
import numpy as np
# from ContourExtraction import Segmentation
WIDTH=640
HIGH=480
# global dataList_c
#
'''形态学操作，包括开闭操作，腐蚀膨胀'''
def open_binary(binary, x, y):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    return dst


def close_binary(binary, x, y):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))
    dst = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    return dst


def erode_binary(binary, x, y):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))
    dst = cv.erode(binary, kernel)
    return dst


def dilate_binary(binary, x, y):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))
    dst = cv.dilate(binary, kernel)
    return dst
'''形态学操作，包括开闭操作，腐蚀膨胀'''
#
# def nothing(x):
#     pass
#
# '''滑动条hsv色彩空间转换，阈值分割'''
# def creatTrackbar():  # creat trackbar to adjust the color threshold.
#     cv.namedWindow("color_adjust")
#     cv.namedWindow("mor_adjust")
#
#     # blue
#     # cv.createTrackbar("hmin", "color_adjust", 0, 255, nothing)
#     # cv.createTrackbar("hmax", "color_adjust", 250, 255, nothing)
#     # cv.createTrackbar("smin", "color_adjust", 0, 255, nothing)
#     # cv.createTrackbar("smax", "color_adjust", 143, 255, nothing)
#     # cv.createTrackbar("vmin", "color_adjust", 255, 255, nothing)
#     # cv.createTrackbar("vmax", "color_adjust", 255, 255, nothing)
#     cv.createTrackbar("hmin", "color_adjust", 0, 255, nothing)
#     cv.createTrackbar("hmax", "color_adjust", 180, 255, nothing)
#     cv.createTrackbar("smin", "color_adjust", 0, 255, nothing)
#     cv.createTrackbar("smax", "color_adjust", 30, 255, nothing)
#     cv.createTrackbar("vmin", "color_adjust", 221, 255, nothing)
#     cv.createTrackbar("vmax", "color_adjust", 255, 255, nothing)
#
#     # red
#     # cv.createTrackbar("hmin", "color_adjust", 0, 255, nothing)
#     # cv.createTrackbar("hmax", "color_adjust", 255, 255, nothing)
#     # cv.createTrackbar("smin", "color_adjust", 3, 255, nothing)
#     # cv.createTrackbar("smax", "color_adjust", 255, 255, nothing)
#     # cv.createTrackbar("vmin", "color_adjust", 245, 255, nothing)
#     # cv.createTrackbar("vmax", "color_adjust", 255, 255, nothing)
#
#     cv.createTrackbar("open", "mor_adjust", 1, 30, nothing)
#     cv.createTrackbar("close", "mor_adjust", 5, 30, nothing)
#     cv.createTrackbar("erode", "mor_adjust", 2, 30, nothing)
#     cv.createTrackbar("dilate", "mor_adjust", 5, 30, nothing)
#
#
# def hsv_change(frame):  # hsv channel separation.
#     hmin = cv.getTrackbarPos('hmin', 'color_adjust')
#     hmax = cv.getTrackbarPos('hmax', 'color_adjust')
#     smin = cv.getTrackbarPos('smin', 'color_adjust')
#     smax = cv.getTrackbarPos('smax', 'color_adjust')
#     vmin = cv.getTrackbarPos('vmin', 'color_adjust')
#     vmax = cv.getTrackbarPos('vmax', 'color_adjust')
#
#     # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # cv.imshow("gray", gray)
#     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#     lower_hsv = np.array([hmin, smin, vmin])
#     upper_hsv = np.array([hmax, smax, vmax])
#     mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
#     return mask
# '''滑动条hsv色彩空间转换，阈值分割'''
#
# '''图像预处理阶段总函数'''
# def read_morphology(cap):  # read cap and morphological operation to get led binary image.
#     ret, frame = cap.read()
#     # frame = cv.flip(frame, 1)
#     # frame = cv.flip(frame, 1)
#
#     open = cv.getTrackbarPos('open', 'mor_adjust')
#     close = cv.getTrackbarPos('close', 'mor_adjust')
#     erode = cv.getTrackbarPos('erode', 'mor_adjust')
#     dilate = cv.getTrackbarPos('dilate', 'mor_adjust')
#     frame = cv.resize(frame, (WIDTH, HIGH), interpolation=cv.INTER_CUBIC)
#
#     mask = hsv_change(frame)
#     # mask = hsvs.extract_hsv(frame)
#     # dst_open = open_binary(mask, open, open)
#     dst_close = close_binary(mask, close, close)
#     dst_erode = erode_binary(dst_close, erode, erode)
#     dst_dilate = dilate_binary(dst_erode, dilate, dilate)
#     cv.circle(frame, (int(WIDTH / 2), int(HIGH / 2)), 2, (255, 0, 255), -1)
#
#     cv.imshow("erode", mask)
#
#     return dst_dilate, frame
# '''图像预处理阶段总函数'''


# def print_hi(name):
#     # 在下面的代码行中使用断点来调试脚本。
#     print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。

'''通道相减、阈值分割'''
def channel_subtract(image):
    # 获得红色通道
    blue_c, green_c, red_c = cv.split(image)
    red_sub_blue = red_c - blue_c
    blue_sub_red = blue_c -red_c
    red_sub_blue = cv.GaussianBlur(red_sub_blue,(3,3),1.5)
    blue_sub_red = cv.GaussianBlur(blue_sub_red,(3,3),1.5)
    normalize_red = cv.normalize(red_sub_blue,0,255,cv.NORM_MINMAX)
    normalize_blue = cv.normalize(blue_sub_red,0,255,cv.NORM_MINMAX)
    _,red_thresh = cv.threshold(normalize_red,0,255,cv.THRESH_OTSU)
    _,blue_thresh = cv.threshold(normalize_blue,0,255,cv.THRESH_OTSU)
    red_dst = open_binary(red_thresh,3,3)
    blue_dst = open_binary(blue_thresh,3,3)
    red_dst = dilate_binary(red_dst,3,3)
    blue_dst = dilate_binary(blue_dst,3,3)
    # cv.Canny(red_dst,red_dst,3,9,3)
    # cv.Canny(blue_dst, blue_dst,3,9,3)
    # red_dst = cv.Canny(red_dst, 50, 150)
    # blue_dst = cv.Canny(blue_dst, 50, 150)
    cv.imshow("red_sub_img", red_dst)
    cv.imshow("blue_sub_img", blue_dst)
'''通道相减、阈值分割'''

# '''边缘、轮廓提取筛选灯条'''
# def find_contours(binary, frame):  # find contours and main screening section
#     # global dataList_c
#     contours, heriachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     length = len(contours)
#     data_list = []
#     first_data = []
#     second_data1 = []
#     second_data2 = []
#     dataList_c = []
#
#     c = 0
#     d = 0
#     if length > 0:
#         # print("---founding---")
#         for i, contour in enumerate(contours):
#             data_dict = dict()
#             # print("countour", contour)
#             area = cv.contourArea(contour)
#             rect = cv.minAreaRect(contour)
#             rx, ry = rect[0]
#             rw = rect[1][0]
#             rh = rect[1][1]
#             z = rect[2]
#
#             coor = cv.boxPoints(rect)
#             x1 = coor[0][0]
#             y1 = coor[0][1]
#             x2 = coor[1][0]
#             y2 = coor[1][1]
#             x3 = coor[2][0]
#             y3 = coor[2][1]
#             x4 = coor[3][0]
#             y4 = coor[3][1]
#
#             # if i >= 1:
#             data_dict["area"] = area
#             data_dict["rx"] = rx
#             data_dict["ry"] = ry
#             data_dict["rh"] = rh
#             data_dict["rw"] = rw
#             data_dict["z"] = z
#             data_dict["x1"] = x1
#             data_dict["y1"] = y1
#             data_dict["x2"] = x2
#             data_dict["y2"] = y2
#             data_dict["x3"] = x3
#             data_dict["y3"] = y3
#             data_dict["x4"] = x4
#             data_dict["y4"] = y4
#             data_list.append(data_dict)
#
#         for i in range(len(data_list)):
#
#             data_rh = data_list[i].get("rh", 0)
#             data_rw = data_list[i].get("rw", 0)
#             data_area = data_list[i].get("area", 0)
#             if (float(data_rh / data_rw) >= 0.2) \
#                     and (float(data_rh / data_rw) <= 4) \
#                     and data_area >= 20:
#                 first_data.append(data_list[i])
#             else:
#                 pass
#         for i in range(len(first_data)):
#
#             c = i + 1
#             while c < len(first_data):
#                 data_ryi = float(first_data[i].get("ry", 0))
#                 data_ryc = float(first_data[c].get("ry", 0))
#                 data_rhi = float(first_data[i].get("rh", 0))
#                 data_rhc = float(first_data[c].get("rh", 0))
#                 data_rxi = float(first_data[i].get("rx", 0))
#                 data_rxc = float(first_data[c].get("rx", 0))
#
#                 if (abs(data_ryi - data_ryc) <= 3 * ((data_rhi + data_rhc) / 2)) \
#                         and (abs(data_rhi - data_rhc) <= 0.2 * max(data_rhi, data_rhc)) \
#                         and (abs(data_rxi - data_rxc) <= (6 / 2) * ((data_rhi + data_rhc) / 2)):
#
#                     second_data1.append(first_data[i])
#                     second_data2.append(first_data[c])
#
#                 c = c + 1
#
#         # for i in range(len(second_data1)):
#         #     data_z1 = second_data1[i].get("z", 0)
#         #     data_z2 = second_data2[i].get("z", 0)
#         #     if abs(data_z1 - data_z2) <= 6:
#         #         third_data1.append(second_data1[i])
#         #         third_data2.append(second_data2[i])
#
#         if len(second_data1):
#             # global dataList_c
#             dataList_c.clear()
#             for i in range(len(second_data1)):
#
#                 rectangle_x1 = int(second_data1[i]["x1"])
#                 rectangle_y1 = int(second_data1[i]["y1"])
#                 rectangle_x2 = int(second_data2[i]["x3"])
#                 rectangle_y2 = int(second_data2[i]["y3"])
#
#                 if abs(rectangle_y1 - rectangle_y2) <= (6 / 2) * (abs(rectangle_x1 - rectangle_x2)):
#
#                     global point1_1x, point1_1y, point1_2x, point1_2y, point1_3x, point1_3y, point1_4x, point1_4y
#                     global point2_1x, point2_1y, point2_2x, point2_2y, point2_3x, point2_3y, point2_4x, point2_4y
#
#                     point1_1x = second_data1[i]["x1"]
#                     point1_1y = second_data1[i]["y1"]
#                     point1_2x = second_data1[i]["x2"]
#                     point1_2y = second_data1[i]["y2"]
#                     point1_3x = second_data1[i]["x3"]
#                     point1_3y = second_data1[i]["y3"]
#                     point1_4x = second_data1[i]["x4"]
#                     point1_4y = second_data1[i]["y4"]
#
#                     point2_1x = second_data2[i]["x1"]
#                     point2_1y = second_data1[i]["y1"]
#                     point2_2x = second_data1[i]["x2"]
#                     point2_2y = second_data1[i]["y2"]
#                     point2_3x = second_data1[i]["x3"]
#                     point2_3y = second_data1[i]["y3"]
#                     point2_4x = second_data1[i]["x4"]
#                     point2_4y = second_data1[i]["y4"]
#
#                     if point1_1x > point2_1x:
#                         pass
#
#                         cv.rectangle(frame, (point2_2x, point2_2y), (point1_4x, point1_4y), (255, 255, 0), 2)
#
#                     else:
#                         point1_1x, point2_1x = point2_1x, point1_1x
#                         point1_2x, point2_2x = point2_2x, point1_2x
#                         point1_3x, point2_3x = point2_3x, point1_3x
#                         point1_4x, point2_4x = point2_4x, point1_4x
#
#                         point1_1y, point2_1y = point2_1y, point1_1y
#                         point1_2y, point2_2y = point2_2y, point1_2y
#                         point1_3y, point2_3y = point2_3y, point1_3y
#                         point1_4y, point2_4y = point2_4y, point1_4y
#                         cv.rectangle(frame, (point2_1x, point2_1y), (point1_3x, point1_3y), (255, 255, 0), 2)
#
#                     cv.putText(frame, "target1:", (rectangle_x2, rectangle_y2 - 5), cv.FONT_HERSHEY_SIMPLEX,
#                                0.5, [255, 255, 255])
#                     center = (int((point2_2x + point1_4x) / 2), int((point2_2y + point1_4y) / 2))
#                     cv.circle(frame, center, 2, (0, 0, 255), -1)  # 画出重心
#
#                     dataList_c.append(center)
#                     high = (rectangle_y1, rectangle_y2)
#                     # dataRange.append(high)
#         else:
#             print("---not find---")
#
#             dataList_c.clear()
#     data_list.clear()
# '''边缘、轮廓提取筛选灯条'''

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':

    cap=cv.VideoCapture('armor.mp4')
    # creatTrackbar()
    # hsvs = Segmentation()
    while(1):
        ret,img = cap.read()
        channel_subtract(img)
        # dst_dilate, frame= #read_morphology(cap)
        # find_contours(dst_dilate, frame)
        # cv.imshow("winName", dst_dilate)
        # cv.imshow("winName1", frame)
        if cv.waitKey(1) == 27:
            break
        # if cv.waitKey(1) == 27:
        #     break
    cap.release()
    cv.destroyAllWindows()