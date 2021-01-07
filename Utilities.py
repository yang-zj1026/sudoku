# import the necessary packages
from imutils.perspective import four_point_transform
import tensorflow as tf
import numpy as np
import imutils
import cv2

def find_puzzle(image, debug=False):
	# convert the image to grayscale and blur it slightly
	# 转变图像为灰度图
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# 高斯滤波用于图像模糊处理
	blurred = cv2.GaussianBlur(gray, (7, 7), 3)

	# apply adaptive thresholding and then invert the threshold map
	# 图像二值化
	thresh = cv2.adaptiveThreshold(blurred, 255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	thresh = cv2.bitwise_not(thresh)

	# check to see if we are visualizing each step of the image
	# processing pipeline (in this case, thresholding)
	if debug:
		cv2.imshow("Puzzle Thresh", thresh)
		cv2.waitKey(0)

	# find contours in the thresholded image and sort them by size in
	# descending order
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE) # 只检测外轮廓；接受二值图而非灰度图；压缩水平垂直方向对角方向的元素，只保留终点坐标
	# cnts是一个list，每个元素都是图像的一个轮廓，即一个ndarray（轮廓上点的集合）
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # 排序，这样先检测大的轮廓

	# initialize a contour that corresponds to the puzzle outline
	puzzleCnt = None

	# loop over the contours
	for c in cnts:
		# approximate the contour
		# 计算轮廓周长
		peri = cv2.arcLength(c, True) 
		# 对轮廓进行多边形近似
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)  

		# if our approximated contour has four points, then we can
		# assume we have found the outline of the puzzle
		if len(approx) == 4:
			puzzleCnt = approx
			break

	# if the puzzle contor is empty then our script could not find
	# the outline of the sudoku puzzle so raise an error
	if puzzleCnt is None:
		raise Exception(("Could not find sudoku puzzle outline. "
			"Try debugging your thresholding and contour steps."))

	# check to see if we are visualizing the outline of the detected
	# sudoku puzzle
	if debug:
		# draw the contour of the puzzle on the image and then display
		# it to our screen for visualization/debugging purposes
		output = image.copy()
		cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
		cv2.imshow("Puzzle Outline", output)
		cv2.waitKey(0)

	# apply a four point perspective transform to both the original
	# image and grayscale image to obtain a top-down birds eye view
	# of the puzzle
	# 四点透视变换变为鸟瞰图
	puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
	warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

	# check to see if we are visualizing the perspective transform
	if debug:
		# show the output warped image (again, for debugging purposes)
		cv2.imshow("Puzzle Transform", puzzle)
		cv2.imshow('Puzzle gray', warped)
		cv2.waitKey(0)

	# return a 2-tuple of puzzle in both RGB and grayscale
	return (puzzle, warped)

def extract_digit(cell, debug=False):
	# apply automatic thresholding to the cell and then clear any
	# connected borders that touch the border of the cell

	thresh1 = cell
	# 二值化
	thresh = cv2.threshold(thresh1, 140, 255,
		cv2.THRESH_BINARY_INV)[1] 

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# if no contours were found than this is an empty cell
	if len(cnts) == 0:
		return None

	# check to see if we are visualizing the cell thresholding step
	if debug:
		cv2.imshow("cell", thresh1)
		cv2.imshow("Cell Thresh", thresh)
		cv2.waitKey(0)
	
	

	# create a mask for the contour
	mask = np.zeros(thresh.shape, dtype="uint8")
	cv2.drawContours(mask, cnts, -1, 255, -1)

	# compute the percentage of masked pixels relative to the total
	# area of the image
	(h, w) = thresh.shape
	percentFilled = cv2.countNonZero(mask) / float(w * h)

	# if less than 5% of the mask is filled then we are looking at
	# noise and can safely ignore the contour
	if percentFilled < 0.005:
		return None

	# apply the mask to the thresholded cell
	digit = cv2.bitwise_and(thresh, thresh, mask=mask)
	digit = modify_img(digit)
	# check to see if we should visualize the masking step
	if debug:
		cv2.imshow("Digit", digit)
		cv2.waitKey(0)

	digit = cv2.resize(digit, (20, 20))
	paddings = tf.constant([[4, 4], [4, 4]])
	digit = np.pad(digit, paddings, "constant")  # padding 补足

	if debug:
		cv2.imshow("Centralized Digit", digit)

	# return the digit to the calling function
	return digit


def modify_img(thresh):
	# 使图像移动至中央
	(h, w) = thresh.shape
	cnt_h = [0] * h
	cnt_w = [0] * w
	for i in range(0, h):
		for j in range(0, w):
			if thresh[i][j] != 0:
				cnt_h[i] += 1
				cnt_w[j] += 1

	h_min = h-1
	h_max = 0
	w_min = w-1
	w_max = 0
	# 获取图像x, y坐标的起始和终止值
	for i in range(h):
		if cnt_h[i] >= 1:
			h_min = min(h_min, i)
			h_max = max(h_max, i)
	for i in range(w):
		if cnt_w[i] >= 1:
			w_min = min(w_min, i)
			w_max = max(w_max, i)

	# 选取范围中较大的一个作为新图像的长度和宽度
	length = max(h_max-h_min, w_max-w_min)

	h_max_new = (h_max + h_min + length) // 2
	h_min_new = (h_max + h_min - length) // 2
	w_max_new = (w_max + w_min + length) // 2
	w_min_new = (w_max + w_min - length) // 2

	new_thresh = thresh[h_min_new:h_max_new + 1, w_min_new:w_max_new + 1]
	if (h_min_new < 0 or h_max_new >= h):
		if (h_min_new < 0):
			new_thresh = thresh[0:h_max_new + 1, w_min_new:w_max_new + 1]
			new_thresh = np.pad(new_thresh, ((-h_min_new, 0), (0, 0)), 'constant', constant_values=0)
			
		else:
			new_thresh = thresh[h_min_new:h, w_min_new:w_max_new + 1]
			new_thresh = np.pad(new_thresh, ((0, h_max_new-h+1), (0, 0)), 'constant', constant_values=0)

	if (w_min_new < 0 or w_max_new >= w):
		if (w_min_new < 0):
			new_thresh = thresh[h_min_new:h_max_new + 1, 0:w_max_new + 1]
			new_thresh = np.pad(new_thresh, ((0, 0), (-w_min_new, 0)), 'constant', constant_values=0)

		else:
			new_thresh = thresh[h_min_new:h_max_new + 1, w_min_new:w]
			new_thresh = np.pad(new_thresh, ((0, 0), (0, w_max_new-w+1)), 'constant', constant_values=0)

	new_thresh = cv2.resize(new_thresh, (28, 28))

	return new_thresh