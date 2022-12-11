# import all suitable libraries
# SRC : https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
# TIYA JAIN 
 
import skimage
import math
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage import data
from skimage.color import *
from skimage.util import random_noise
from skimage import feature
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_otsu


# input the image from system
inputImage=skimage.io.imread('assignment1\canny_image.jpg')
# inputImage=data.astronaut()
inputImage=rgb2gray(inputImage)
# plt.imshow(inputImage)
image_1= feature.canny(inputImage,3)



#determining lower and upper thresholds 
v = 150
#  apply automatic Canny edge detection 
lower = int(0.1*v)
upper = int(0.15*v)

# upper=threshold_otsu(inputImage)
# lower=0.5*upper



# convolution
def convolution(image,kernel):

    image_h,image_w= image.shape
    kernel_h,kernel_w = kernel.shape

    h1=kernel_h//2
    w1=kernel_w//2

    filtered_img=np.zeros(image.shape)

    for i in range(h1,image_h-h1):
        for j in range(w1,image_w-w1):
            sum=0

            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum=sum+kernel[m][n]*image[i-h1+m][j-w1+n]
            
            filtered_img[i][j]=sum

    return filtered_img


# gaussian filter
def GaussianBlurImage(m,n,sigma):
    gaussian_filter = np.zeros((m,n))
    m = m//2
    n = n//2
        
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = int(sigma)*(2*np.pi)**2
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaussian_filter[x+m, y+n] = (1/x1)*x2
    
    return gaussian_filter


def calc_grad(inputImage):
    # the kernels
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Tx=convolution(inputImage,Kx)
    Ty=convolution(inputImage,Ky)
    
    mag=np.sqrt(np.square(Tx)+np.square(Ty))
    mx=mag.max()
    mag=mag*255
    mag/=mx

    D=np.arctan2(Ty,Tx)
    # plt.imshow(mag)
    return (mag,D)
    

def non_max_sup(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
    PI=180
    for i in range(1, M-1):
            for j in range(1, N-1):
                try:
                    q,r = 255,255

                   # angle 0
                    if (0 <= angle[i, j] <= PI/8) or ( 15*PI/8 <= angle[i, j] <= PI):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    # angle 45
                    elif (PI/8 <= angle[i, j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i, j] >= q) and (img[i, j] >= r):
                        Z[i, j] = img[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass
    # plt.imshow(Z)
    return Z


def threshold(img,lowThreshold, highThreshold):

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int64)

        wk = np.int64(25)
        stg = np.int64(255)

        stg_i, stg_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        wk_i, wk_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[stg_i, stg_j] = stg
        res[wk_i, wk_j] = wk

        return (res)


def hysteresis(img,wk=25,stg=255):
# four loops of traversals
        M, N = img.shape
       
        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == wk):
                    try:
                        if ((img[i+1, j-1] == stg) or (img[i+1, j] == stg) or (img[i+1, j+1] == stg)
                            or (img[i, j-1] == stg) or (img[i, j+1] == stg)
                            or (img[i-1, j-1] == stg) or (img[i-1, j] == stg) or (img[i-1, j+1] == stg)):
                            img[i, j] = stg
                        else:
                            img[i, j] = 0
                    except IndexError as e: # Ignore the exception and try the next type.
                        pass
        for i in range(M-1,1):
            for j in range(N-1,1):
                if (img[i,j] == wk):
                    try:
                        if ((img[i+1, j-1] == stg) or (img[i+1, j] == stg) or (img[i+1, j+1] == stg)
                            or (img[i, j-1] == stg) or (img[i, j+1] == stg)
                            or (img[i-1, j-1] == stg) or (img[i-1, j] == stg) or (img[i-1, j+1] == stg)):
                            img[i, j] = stg
                        else:
                            img[i, j] = 0
                    except IndexError as e: # Ignore the exception and try the next type.
                        pass

        for i in range(M-1,1):
            for j in range(1, N-1):
                if (img[i,j] == wk):
                    try:
                        if ((img[i+1, j-1] == stg) or (img[i+1, j] == stg) or (img[i+1, j+1] == stg)
                            or (img[i, j-1] == stg) or (img[i, j+1] == stg)
                            or (img[i-1, j-1] == stg) or (img[i-1, j] == stg) or (img[i-1, j+1] == stg)):
                            img[i, j] = stg
                        else:
                            img[i, j] = 0
                    except IndexError as e: # Ignore the exception and try the next type.
                        pass

        for i in range(1, M-1):
            for j in range(N-1,1):
                if (img[i,j] == wk):
                    try:
                        if ((img[i+1, j-1] == stg) or (img[i+1, j] == stg) or (img[i+1, j+1] == stg)
                            or (img[i, j-1] == stg) or (img[i, j+1] == stg)
                            or (img[i-1, j-1] == stg) or (img[i-1, j] == stg) or (img[i-1, j+1] == stg)):
                            img[i, j] = stg
                        else:
                            img[i, j] = 0
                    except IndexError as e:# Ignore the exception and try the next type.
                        pass
        
        return img

