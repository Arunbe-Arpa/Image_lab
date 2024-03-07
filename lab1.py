#implementation of convolution

import numpy as np
import cv2

img = cv2.imread("Lenna.png",cv2.IMREAD_GRAYSCALE)

cv2.imshow("Inputimg",img)
cv2.waitKey(0)

kernel_x = np.array([[-1,0,1],
                   [-1,0,1],
                   [-1,0,1]])

image_r = img.shape[0]
image_c = img.shape[1]

kernel_r = kernel_x.shape[0]
kernel_c = kernel_x.shape[1]

pad_r = kernel_r // 2
pad_c = kernel_c // 2

out = np.zeros((img.shape[0],img.shape[1]))  #output array

for x in range (pad_r ,image_r - pad_r ):
    for y in range (pad_c ,image_c - pad_c ):
        sum = 0
        for m in range ( -pad_r , pad_r+1):
            for n in range (-pad_c , pad_c +1 ):
                sum += img[x-m][y-n] * kernel_x[pad_r+m][pad_c+n]

        out[x][y] = sum


cv2.waitKey(0)      
cv2.imshow('output image',out)
print(out)
cv2.normalize(out,out, 0, 255, cv2.NORM_MINMAX)
out = np.round(out).astype(np.uint8)
print(out)
cv2.imshow('normalised output image',out)


cv2.waitKey(0)
cv2.destroyAllWindows()


#np.array([[-1,-1,1],
 #                  [0,0,0],
 #                  [1,1,1]])