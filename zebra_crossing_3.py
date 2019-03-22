import cv2
import numpy as np
import math


def detect(img):
	img = cv2.resize(img, (800,600))
	grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	mask_white = cv2.inRange(grey_image, 200, 255)
	tmp = cv2.bitwise_and(img, img, mask = mask_white)

	edges = cv2.Canny(tmp, 500, 500)
	# return edges
	lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 10,minLineLength = 50,maxLineGap = 10)
	# print(len(lines))
	if lines is None:
		return (0,0, False)

	lines2 = []
	for l in lines:
		line = l[0]
		x1,y1 = line[0],line[1]
		x2,y2 = line[2],line[3]
		# cv2.line(img, (x1,y1),(x2,y2), (255,0,0), 2)

		if x1 - x2 == 0:
			lines2.append([x1,y1,x2,y2])
			continue

		slope = (y1-y2)/(x1-x2)
		# print(slope)

		if not 0<=abs(slope)<=1.732:
			lines2.append([x1,y1,x2,y2])

	parallel_lines = 0
	# print(lines2)
	for x1,y1,x2,y2 in lines2:
		cv2.line(img, (x1,y1),(x2,y2), (0,0,255), 2)
		parallel_lines += 1

	# print(np.median(lines2, axis = 0))
	xm1,xm2,ym1,ym2 = map(int,np.median(lines2, axis = 0))
	xm = (xm1 + xm2)//2
	ym = (ym1 + ym2)//2
	# cv2.circle(img, (xm,ym), 3, (0,0,255))
	# cv2.rectangle(img, (xm - 200, ym - 100), (xm + 200, ym + 100), (255,255,0), thickness = 2)

	#print(parallel_lines)
	if parallel_lines >= 10:
		#print("zebra crossing present")
		return (xm,ym, True)
	else:
		return (xm,ym, False)
