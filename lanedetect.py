
import numpy as np
import cv2
import math

lintercept, rintercept = [], []
lslope, rslope = [], []

width, height = 800,600
center = (width//2-65, height - height//3)

turn = 0
msg = "pw"

def drawlines(m1,c1,m2,c2, img):
    x,y = center
    y1 = 200
    y2 = height
    x1 = int((y1 - c1)/m1)
    x2 = int((y2 - c1)/m1)
    dl = ((abs(y - m1*x - c1))/math.sqrt(1 + m1**2))

    y3 = 200
    y4 = height
    x3 = int((y3 - c2)/m2)
    x4 = int((y4 - c2)/m2)
    dr = ((abs(y - m2*x - c2))/math.sqrt(1 + m2**2))

    print(dl,dr)

    global turn
    diff = dl - dr
    if diff > 30:
        turn = -1
    elif diff < -30:
        turn = 1
    else:
        turn = 0

    points = np.array([[x1,y1],[x2,y2],[x4,y4],[x3,y3]], dtype = np.int32)
    # cv2.fillConvexPoly(img, points, (225,200,135))
    cv2.line(img, (x1,y1),(x2,y2), (0,0,255), 5)
    cv2.line(img, (x3,y3),(x4,y4), (0,0,255), 5)
    cv2.circle(img, center, 3, (0,255,255))
    cv2.putText(img, f"turn = {turn}", (width//3,height//3), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,255), thickness = 2)


def roi(img):
    vertices = np.array([[0,600],[0,300],[100,300],[700,300],[800,400],[800,600]], np.int32)
    mask = np.zeros_like(img)
    # print(mask)
    cv2.fillConvexPoly(mask, vertices, (255,255,255))
    # return mask
    masked = cv2.bitwise_and(img,mask)
    return masked


def detect(img):
    global msg

    img = cv2.resize(img, (width, height))

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    yel_min = np.array([15, 125, 140],np.uint8)
    yel_max = np.array([30, 255, 255],np.uint8)
    mask_yellow = cv2.inRange(hsv_img, yel_min, yel_max)
    tmp = cv2.bitwise_and(img, img, mask = mask_yellow)
    mask_white = cv2.inRange(grey_image, 200, 255)
    tmp2 = cv2.bitwise_and(img, img, mask = mask_white)
    tmp = cv2.bitwise_or(tmp, tmp2)
    # return tmp

    # roi_img = tmp[:,:]
    # return roi_img
    # mask = np.zeros_like(tmp)
    # mask[width//4:width-width//4] = roi_img
    edges = cv2.Canny(tmp, 500, 500)
    # return edges

    lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 30,minLineLength = 5,maxLineGap = 50)

    if lines is None:
        msg = "pw"
        # print("no lines")s
        return img
    for l in lines:
        line = l[0]
        x1,y1 = line[0],line[1]
        x2,y2 = line[2],line[3]
        if x2-x1 == 0:
            continue
        slope = (y1-y2)/(x1-x2)
        if 0.364<=abs(slope)<=5:
            # cv2.line(img, (x1,y1),(x2,y2), (255,0,0), 2)
            intercept = y1 - slope * x1
            if slope > 0:       #right
                rintercept.append(intercept)
                rslope.append(slope)
            else:               #left
                lintercept.append(intercept)
                lslope.append(slope)
    try:
        m1 = np.mean(lslope[-30:])
        c1 = int(np.mean(lintercept[-30:]))
        m2 = np.mean(rslope[-30:])
        c2 = int(np.mean(rintercept[-30:]))
        drawlines(m1,c1,m2,c2, img)
        # return edges
        msg = "fw"
    except:
        # print("NaN error")
        pass
    

    #msg = "image"
    return img

if __name__ == '__main__':
    img = cv2.imread("img3.jpg")
    img = detect(img)
    while True:
        cv2.imshow("winname", img)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
