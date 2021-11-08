import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
问题1 通过计算一维傅里叶变换实现图像二维快速傅里叶变换（10分）
实现一个函数 F=dft2D(f), 其中f是一个灰度源图像，F是其对应的二维快速傅里叶变换(FFT)图像. 
具体实现要求按照课上的介绍通过两轮一维傅里叶变换实现。也就是首先计算源图像每一行的一维傅里
叶变换，然后对于得到的结果计算其每一列的一维傅里叶变换。
如果实现采用MATLAB, 可以直接调用函数fft计算一维傅里叶变换。如果采用其他语言，请选择并直接
调用相应的一维傅里叶变换函数。
"""

def FFT(x):
    #对二维矩阵的每行做一次傅里叶变换
    N = x.shape[1]
    if N & (-N) != N:
        raise ValueError("Size of x invalid!")
    elif N <= 8:
        #直接调用一维傅里叶变换函数np.fft.fft
        return np.array([np.fft.fft(x[i,:]) for i in range(x.shape[0])])
    else:
        X_even = FFT(x[:,::2])
        X_odd = FFT(x[:,1::2])
        factor = np.array([np.exp(-2j * np.pi * np.arange(N) / N) for i in range(x.shape[0])])
        return np.hstack([X_even + np.multiply(factor[:,:int(N/2)],X_odd),
                               X_even + np.multiply(factor[:,int(N/2):],X_odd)])

def dft2D(f):
    #调用两次 FFT，完成二维傅里叶变换
    return FFT(FFT(f).T).T

def FFT_SHIFT(f):
    #中心化
    M,N = f.shape
    M = int(M/2)
    N = int(N/2)
    return np.vstack((np.hstack((f[M:,N:],f[M:,:N])),np.hstack((f[:M,N:],f[:M,:N]))))

if __name__ == '__main__':
    #读入图像
    img1 = cv.imread('house.tif',cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('lena_gray_512.tif',cv.IMREAD_GRAYSCALE)

    #计算结果
    fft1 = abs(FFT_SHIFT(dft2D(img1)))
    fft2 = abs(FFT_SHIFT(dft2D(img2)))

    # 结果展示
    plt.title('house')
    plt.imshow(img1,cmap=plt.get_cmap('gray'))
    plt.show()
    plt.title('house dft2D')
    plt.imshow(np.log(1 + fft1),cmap=plt.get_cmap('gray'))
    plt.show()
    plt.title('lena_gray_512')
    plt.imshow(img2, cmap=plt.get_cmap('gray'))
    plt.show()
    plt.title('lena_gray_512 dft2D')
    plt.imshow(np.log(1 + fft2), cmap=plt.get_cmap('gray'))
    plt.show()