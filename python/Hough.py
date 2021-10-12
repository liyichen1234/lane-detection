import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = '/data01/lyc/CV-Ex1/test1/Czech/images/Czech_000004.jpg'
path2 = '/data01/lyc/CV-Ex1/benchmark_velocity_supp/supp_img'

# 方法一：Hough检测+Canny检测（简单方便），实用性较差（无法解决弯道的问题）
def Hough(img,num):
    # img = cv2.imread(path2)
    img_size = (img.shape[1],img.shape[0])
    # 复制图片
    test_img = np.copy(img)
    # 转为灰度图
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
    # 高斯模糊
    blur_img = cv2.GaussianBlur(gray_img,(5,5),0)
    # edge_img = cv2.Canny(blur_img,40,100)
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2] 
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # print(ignore_mask_color)
    # 区域的设定方法（如何自动寻找）添加掩膜
    vertices = np.array([[(0,img.shape[0]),(img.shape[1]//2-100,350),(img.shape[1]//2+400,350),(img.shape[1],img.shape[0])]],dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_img = cv2.Canny(cv2.cvtColor(cv2.bitwise_and(img, mask),cv2.COLOR_RGB2GRAY),40,100)
    # masked_img = cv2.bitwise_and(img, mask)
    # cv2.imwrite('1.jpg',masked_img)
    # cv2.imwrite('2.jpg',edge_img)
    def extend_lines(x1,y1,x2,y2,length,width):
        k = (y2-y1)/(x2-x1)
        b = y1-k*x1
        
    def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
        # lines = sorted(lines,key=lambda x:x[0][0])[1:-1]
        for line in lines:
            for x1,y1,x2,y2 in line:
                # 斜率的筛选
                if x2-x1 != 0 and abs((y2-y1)/(x2-x1)) > 0.5:
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness)


    def draw_lines_n(img, lines, color=[255,0,0], thickness=2):
        left_lines_x = []
        left_lines_y = []
        right_lines_x = []
        right_lines_y = []
        line_y_max = 0
        line_y_min = 999
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1 > line_y_max:
                    line_y_max = y1
                if y2 > line_y_max:
                    line_y_max = y2
                if y1 < line_y_min:
                    line_y_min = y1
                if y2 < line_y_min:
                    line_y_min = y2
                if x2-x1 == 0:
                    k = 0
                else:
                    k = (y2 - y1)/(x2 - x1)
                if k < -0.3:
                    left_lines_x.append(x1)
                    left_lines_y.append(y1)
                    left_lines_x.append(x2)
                    left_lines_y.append(y2)
                elif k > 0.3:
                    right_lines_x.append(x1)
                    right_lines_y.append(y1)
                    right_lines_x.append(x2)
                    right_lines_y.append(y2)
    # 最小二乘直线拟合
        left_line_k, left_line_b = np.polyfit(left_lines_x, left_lines_y, 1)
        right_line_k, right_line_b = np.polyfit(right_lines_x, right_lines_y, 1)

        # 根据直线方程和最大、最小的y值反算对应的x
        cv2.line(img,
                (int((line_y_max - left_line_b)/left_line_k), line_y_max),
                (int((line_y_min - left_line_b)/left_line_k), line_y_min),
                color, thickness)
        cv2.line(img,
                (int((line_y_max - right_line_b)/right_line_k), line_y_max),
                (int((line_y_min - right_line_b)/right_line_k), line_y_min),
                color, thickness)


    lines = cv2.HoughLinesP(masked_img, rho=2, theta=np.pi/180,threshold=120, minLineLength=20, maxLineGap=5)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    final_img = cv2.addWeighted(line_img,0.8,img,1.,0.)
    cv2.imwrite('hough/'+num+'.jpg',final_img)
    # cv2.imwrite('1.jpg',edge_img)

for i in range(10,100):
    npath = path2+'/00'+str(i)+'.jpg'
    print(npath)
    img = cv2.imread(npath)
    Hough(img,str(i))
    