import cv2 as cv 
import numpy as np
import mapper
img = cv.imread('Photos/receipt.jpg')
#cv.imshow('cat', img)
#resized = cv.resize(img, (200,200))
#cv.imshow('resi', resized)
blank = np.zeros(img.shape, dtype='uint8')
#cv.imshow('blank', blank)



#grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('grayed', gray)

#blurring
blurred = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)
cv.imshow('blur', blurred)

#canny edge
canny = cv.Canny(blurred, 125, 175)
cv.imshow('edges', canny)

#contours
contours, heirarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contours detected')
contours = sorted(contours, key = cv.contourArea, reverse = True)



for curve in contours:
    #find the perimeter and approximate the contour
    perimeter = cv.arcLength(curve, True)
    approx = cv.approxPolyDP(curve, 0.02*perimeter, True)

    #if the approximate returns 4 points, which is basically a square or a rectangle, exit the loop
    if len(approx) == 4:
        result = approx
        break

cv.drawContours(img, [result], -1, (0,0,255), 6)
cv.imshow('contours drawn', img)

approx = mapper.mapp(result)

points = np.float32([[0,0],[400,0],[400,400],[0,400]])

output = cv.getPerspectiveTransform(approx, points)
dist = cv.warpPerspective(img, output, (400,400))

cv.imshow('Scanned', dist)


cv.waitKey(0)