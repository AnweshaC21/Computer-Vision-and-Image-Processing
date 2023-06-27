import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread("img1.jpg")
img2 = cv2.imread("img3.jpg")
img = img1
points = []

def printCoordinate(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x,y), 3, (255,255,255), -1)
        strXY = '(' + str(x) + ',' + str(y) + ')'
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, strXY, (x+10,y-10), font, 1, (255,255,255))
        cv2.imshow('image',img)
        points.append((x,y))

cv2.imshow('image', img)
cv2.setMouseCallback('image', printCoordinate)
cv2.waitKey()
cv2.destroyAllWindows()
points_img1 = points
print("Points from Img 1: ", points_img1)

img = img2
points = []
cv2.imshow('image', img)
cv2.setMouseCallback('image', printCoordinate)
cv2.waitKey()
cv2.destroyAllWindows()
points_img2 = points
print("Points from Img 2: ", points_img2)

'''
points_img1 = [(676, 81), (676, 100), (309, 157), (1063, 272), (680, 240), (447, 438), (529, 449), (961, 455), (1139, 469), (395, 473), (416, 533), (803, 543), (1179, 591), (320, 595), (378, 636), (1126, 643), (1206, 679), (279, 681), (424, 733), (1213, 734), (1239, 787)]
points_img2 =  [(616, 51), (615, 72), (245, 130), (997, 248), (618, 212), (336, 412), (419, 422), (853, 431), (1028, 444), (285, 448), (305, 507), (691, 515), (1066, 566), (209, 570), (262, 611), (1012, 621), (1093, 656), (165, 658), (311, 709), (1096, 708), (1120, 762)]
'''

A = []

for i in range(len(points_img1)):
    x, y = points_img1[i]
    x_, y_ = points_img2[i]

    row1 = [-x, -y, -1, 0, 0, 0, x_ * x, x_ * y, x_]
    row2 = [0, 0, 0, -x, -y, -1, y_ * x, y_ * y, y_]
    A.append(row1)
    A.append(row2)

A = np.array(A)
print('\nA =\n', A)

u, s, v = np.linalg.svd(A)
H = v[-1].reshape(3, 3)
H /= H[2][2]

print('H matrix: \n', H)

H_, mask = cv2.findHomography(np.array(points_img1), np.array(points_img2), cv2.RANSAC, 2)
print('H_: \n', H_)

# plotting corresponding points (x vs x')
x1s = [x[0] for x in points_img1]
y1s = [x[1] for x in points_img1]
plt.plot(x1s, y1s, 'g')
x2s = [x[0] for x in points_img2]
y2s = [x[1] for x in points_img2]
plt.plot(x2s, y2s, 'b')
plt.xlabel('x')
plt.ylabel('x\'')
plt.legend(['Image 1 points', 'Image 2 points'])
plt.show()


# Map new points using H
points = []
img = img1
cv2.imshow('image', img)
cv2.setMouseCallback('image', printCoordinate)
cv2.waitKey()
cv2.destroyAllWindows()
test_points1 = points

# x' = H * x

cv2.imshow('image', img2)
for i in range(len(test_points1)):
    x, y = test_points1[i]
    a = [x, y, 1]
    a_ = np.matmul(H, np.transpose(a))
    x_, y_, z_ = a_
    x_ = int(x_/z_)
    y_ = int(y_/z_)
    print(x_, y_)
    strXY = '(' + str(x_) + ',' + str(y_) + ')'
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(img, strXY, (x + 10, y - 10), font, 1, (255, 255, 255))
    cv2.circle(img2, (x_, y_), 10, (255, 255, 255), 5)
    cv2.imshow('image', img2)
    cv2.waitKey()

cv2.destroyAllWindows()
