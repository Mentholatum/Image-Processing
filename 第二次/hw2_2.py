from hw2_1 import *

'''
问题2 图像二维快速傅里叶逆变换（10分）
实现一个函数f=idft2D(F), 其中F是一个灰度图像的傅里叶变换，f是其对应的二维
快速傅里叶逆变换(IFFT)图像，也就是灰度源图像. 具体实现要求按照课上的介绍通
过类似正向变换的方式实现。
'''

def idft2D(F):
    #针对傅里叶变换图像F，作逆变换
    M = np.size(F,0)
    N = np.size(F,1)
    F = np.conj(F)
    temp = np.conj(dft2D(F))/(M*N)
    f = temp
    return f


if __name__ == '__main__':
    # 读入图像
    img1 = cv.imread('house.tif',cv.IMREAD_GRAYSCALE)
    fft1 = abs(FFT_SHIFT(idft2D(img1)))
    img2 = cv.imread('lena_gray_512.tif',cv.IMREAD_GRAYSCALE)
    fft2 = abs(FFT_SHIFT(idft2D(img2)))

    #结果展示
    plt.title('house')
    plt.imshow(img1,cmap=plt.get_cmap('gray'))
    plt.show()
    plt.title('house idft2D')
    plt.imshow(np.log(1 + fft1),cmap=plt.get_cmap('gray'))
    plt.show()
    plt.title('lena gray')
    plt.imshow(img2, cmap=plt.get_cmap('gray'))
    plt.show()
    plt.title('lena gray idft2D')
    plt.imshow(np.log(1 + fft2), cmap=plt.get_cmap('gray'))
    plt.show()