import cv2
import numpy as np

img1 = cv2.imread("chess_pattern.jpeg")
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

'''
points_img1 = [(886, 563), (799, 665), (755, 534), (894, 657), (712, 624), (833, 477), (764, 682), (736, 473), (946, 517)]
'''

C = []

for i in range(len(points_img1)):
    x, y = points_img1[i]
    X, Y, Z = map(int, input(f'Enter value for 3D point {i+1}: ').split())

    row1 = [-X, -Y, -Z, -1, 0, 0, 0, 0, x*X, x*Y, x*Z, x]
    row2 = [0, 0, 0, 0, -X, -Y, -Z, -1, y*X, y*Y, y*Z, y]
    C.append(row1)
    C.append(row2)

C = np.array(C)
print('\nC =\n', C)

u, s, v = np.linalg.svd(C)
P = v[-1].reshape(3, 4)
P /= P[2][3]

print('\nP matrix: \n', P)

# Verification
w_pt = [4, 5, 0, 1]
x_i, y_i, z_i = np.transpose(np.dot(P, w_pt))
print(x_i, y_i, z_i)
x_i = int(x_i/z_i)
y_i = int(y_i/z_i)

cv2.imshow('image', img1)
strXY = '(' + str(x_i) + ',' + str(y_i) + ')'
font = cv2.FONT_HERSHEY_PLAIN
cv2.putText(img1, strXY, (x_i + 10, y_i - 10), font, 1, (255, 255, 255))
cv2.circle(img1, (x_i, y_i), 10, (255, 255, 255), 5)
cv2.imshow('image', img1)
cv2.waitKey()

cv2.destroyAllWindows()