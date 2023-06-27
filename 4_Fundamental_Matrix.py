import cv2
import numpy as np

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


# F calculation
B = []

for i in range(len(points_img1)):
    x, y = points_img1[i]
    x_, y_ = points_img2[i]

    row = [x_*x, x_*y, x_, x*y_, y*y_, y_, x, y, 1]
    B.append(row)

B = np.array(B)
print('\nB =\n', B)

u, s, v = np.linalg.svd(B)
F = v[-1].reshape(3,3)
F /= F[2][2]

print('\nFundamental Matrix: \n', F)

# verification (x' * F * x_t = 0)
print('\nValues of x\'*F*x_t for each set of corresponding points:')

for i in range(len(points_img1)):
    x, y = points_img1[i]
    x_, y_ = points_img2[i]

    mat = np.matmul(np.matmul([x_, y_, 1], F), np.transpose([x, y, 1]))
    print(mat)
