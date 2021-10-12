import cv2
import PIL
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

# from numpy.core.records import array
path = '/data01/lyc/CV-Ex1/test1/Czech/images/Czech_000004.jpg'
path2 = '/data01/lyc/CV-Ex1/benchmark_velocity_supp/supp_img/0047.jpg'

# 方法一：Hough检测+Canny检测（简单方便），实用性较差（无法解决弯道的问题）
'''
img = cv2.imread(path2)
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
cv2.imwrite('1.jpg',masked_img)
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
cv2.imwrite('2.jpg',final_img)
# cv2.imwrite('1.jpg',edge_img)
'''
# 方法二：使用俯瞰视角，需要事先对摄像机进行畸变矫正效果会更好。可以很好的解决弯道的问题，但是面对较暗的情况和行道线过于稀疏就难以获得正确答案。

origin_img = cv2.imread(path2)
test_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

def perspective_transform(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def subplot(rows, cols, imgs):
    nums = rows * cols
    for i in range(1, nums+1):
        plt.subplot(rows, cols, i)

# left_top to left_bottom（根据相机特点手动设置的超参）
corners = [(603, 445), (677, 445), (1105, 720), (205, 720)]
# 透视变换
wrap_offset = 150
src_corners = [(603, 445), (677, 445), (1105, test_img.shape[0]), (205, test_img.shape[0])]
dst_corners = [(205 + wrap_offset, 0), (1105 - wrap_offset, 0), (1105 - wrap_offset, test_img.shape[0]), (205 + wrap_offset, test_img.shape[0])]
M = cv2.getPerspectiveTransform(np.float32(src_corners), np.float32(dst_corners))
wrap_img= perspective_transform(test_img, M)
subplot(1, 2, [test_img, wrap_img])

def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

ksize = 9 # Choose a larger odd number to smooth gradient measurements
gradx = abs_sobel_thresh(wrap_img, orient='x', sobel_kernel=3, thresh=(20, 255))
mag_binary = mag_thresh(wrap_img, sobel_kernel=3, mag_thresh=(30, 100))
dir_binary = dir_threshold(wrap_img, sobel_kernel=15, thresh=(0.7, 1.3))

# Plot the result
# f, axs = plt.subplots(2, 2, figsize=(16, 9))
# f.tight_layout()
# axs[0, 0].imshow(wrap_img)
# axs[0, 0].set_title('Original Image', fontsize=18)
# axs[0, 1].imshow(gradx, cmap='gray')
# axs[0, 1].set_title('Sobel_x_filter', fontsize=18)
# axs[1, 0].imshow(dir_binary, cmap='gray')
# axs[1, 0].set_title('Dir_threshold', fontsize=18)
# axs[1, 1].imshow(mag_binary, cmap='gray')
# axs[1, 1].set_title('Mag_threshold', fontsize=18)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.savefig('3.jpg')

def r_select(img, thresh=(200, 255)):
    R = img[:,:,0]
    binary = np.zeros_like(R)
    binary[(R > thresh[0]) & (R <= thresh[1])] = 1
    return binary

def color_mask(hsv,low,high):
    # Return mask from HSV 
    mask = cv2.inRange(hsv, low, high)
    return mask

def apply_color_mask(hsv,img,low,high):
    # Apply color mask to image
    mask = cv2.inRange(hsv, low, high)
    res = cv2.bitwise_and(img,img, mask= mask)
    return res

def apply_yellow_white_mask(img):
    image_HSV = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    yellow_hsv_low  = np.array([ 0,  100,  100])
    yellow_hsv_high = np.array([ 80, 255, 255])
    white_hsv_low  = np.array([ 0,   0,   160])
    white_hsv_high = np.array([ 255,  80, 255])   
    mask_yellow = color_mask(image_HSV,yellow_hsv_low,yellow_hsv_high)
    mask_white = color_mask(image_HSV,white_hsv_low,white_hsv_high)
    mask_YW_image = cv2.bitwise_or(mask_yellow,mask_white)
    return mask_YW_image

r_binary = r_select(wrap_img, thresh=(220, 255))
yw_binary = apply_yellow_white_mask(wrap_img)
# Plot the result
# f, axs = plt.subplots(1, 2, figsize=(16, 9))
# f.tight_layout()
# axs[0].imshow(r_binary, cmap='gray')
# axs[0].set_title('R filter', fontsize=18)
# axs[1].imshow(yw_binary, cmap='gray')
# axs[1].set_title('Yellow white filter', fontsize=18)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.savefig('4.jpg')

def hls_select(img, channel='S', thresh=(90, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel == 'S':
        X = hls[:, :, 2]
    elif channel == 'H':
        X = hls[:, :, 0]
    elif channel == 'L':
        X = hls[:, :, 1]
        return

    binary_output = np.zeros_like(X)
    binary_output[(X > thresh[0]) & (X <= thresh[1])] = 1
    return binary_output

l_binary = hls_select(wrap_img, channel='L', thresh=(100, 200))
s_binary = hls_select(wrap_img, channel='S', thresh=(100, 255))
h_binary = hls_select(wrap_img, channel='H', thresh=(100, 255))
# f, axs = plt.subplots(2, 2, figsize=(16, 9))
# f.tight_layout()
# axs[0, 0].imshow(wrap_img)
# axs[0, 0].set_title('Original Image', fontsize=18)
# axs[0, 1].imshow(h_binary, cmap='gray')
# axs[0, 1].set_title('H channal filter', fontsize=18)
# axs[1, 0].imshow(s_binary, cmap='gray')
# axs[1, 0].set_title('S channal filter', fontsize=18)
# axs[1, 1].imshow(l_binary, cmap='gray')
# axs[1, 1].set_title('L channal filter', fontsize=18)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.savefig('5.jpg')


def combine_filters(img):
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 255))
    l_binary = hls_select(img, channel='L', thresh=(100, 200))
    s_binary = hls_select(img, channel='S', thresh=(100, 255))
    yw_binary = apply_yellow_white_mask(wrap_img)
    yw_binary[(yw_binary !=0)] = 1
    combined_lsx = np.zeros_like(gradx)
    combined_lsx[((l_binary == 1) & (s_binary == 1) | (gradx == 1) | (yw_binary == 1))] = 1
    return combined_lsx

binary = combine_filters(wrap_img)
# f, axs = plt.subplots(1, 2, figsize=(16, 9))
# f.tight_layout()
# axs[0].imshow(wrap_img)
# axs[0].set_title('Original', fontsize=18)
# axs[1].imshow(binary, cmap='gray')
# axs[1].set_title('combine filters', fontsize=18)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.savefig('6.jpg')

def find_line_fit(img, nwindows=9, margin=100, minpix=50):
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    out_img = np.dstack((img, img, img)) * 255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    window_height = np.int(img.shape[0]/nwindows)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # to plot
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit, out_img

# Generate x and y values for plotting
def get_fit_xy(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return left_fitx, right_fitx, ploty

left_fit, right_fit, out_img = find_line_fit(binary)
left_fitx, right_fitx, ploty = get_fit_xy(binary, left_fit, right_fit)

# fig = plt.figure(figsize=(16, 9))
# plt.imshow(out_img)
# plt.plot(left_fitx, ploty, color='white', linewidth=3.0)
# plt.plot(right_fitx, ploty, color='white',  linewidth=3.0)
# plt.xlim(0, 1280)
# plt.ylim(720, 0)
# plt.savefig('7.jpg')

def project_back(wrap_img, origin_img, left_fitx, right_fitx, ploty, M):
    warp_zero = np.zeros_like(wrap_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = perspective_transform(color_warp, M)
    # Combine the result with the original image
    result = cv2.addWeighted(origin_img, 1, newwarp, 0.3, 0)
    return result

M = cv2.getPerspectiveTransform(np.float32(dst_corners), np.float32(src_corners))
result = project_back(binary, test_img, left_fitx, right_fitx, ploty, M)
cv2.imwrite('8.jpg',cv2.cvtColor(result, cv2.COLOR_BGR2RGB))