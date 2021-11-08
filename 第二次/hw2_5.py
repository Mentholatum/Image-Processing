import matplotlib.pyplot as plt

from hw2_1 import *

"""
选做题测试更多图像的二维快速傅里叶变换(10 分)
计 算 其他5 幅图像的二维快速傅里叶变换： house.tif, house02.tif, lena_gray_512.tif,
lunar_surface.tif, characters_test_pattern.tif。注意，有些图像的尺寸不是2 的整数次幂，需要
进行相应的像素填补处理。如果图像有多个通道可以选择其中的一个通道进行计算。
"""

def ImgChannel(f):
    #图像通道判断和获取
    sp = f.shape
    if (sp != 1):
        img = f[:,:, 0]#读取蓝色通道
        return img
    else:
        return f

def ImgResize(f):
    #图像像素判断和修改
    N = f.shape[1]
    if N & (-N) != N:
        # 拟定修改尺寸
        width = 512
        height = 512
        dim = (width, height)
        img = cv.resize(f, dim, interpolation=cv.INTER_AREA)
        return img
    else:
        return f


if __name__ == '__main__':
    #读取图像并作修改
    img_1 = cv.imread('Characters_test_pattern.tif')
    img_1 = ImgResize(img_1)
    img_1 = ImgChannel(img_1)
    fft_1 = abs(FFT_SHIFT(dft2D(img_1)))

    img_2 = cv.imread('house02.tif')
    img_2 = ImgResize(img_2)
    img_2 = ImgChannel(img_2)
    fft_2 = abs(FFT_SHIFT(dft2D(img_2)))

    img_3 = cv.imread('rose512.tif')
    img_3 = ImgResize(img_3)
    img_3 = ImgChannel(img_3)
    fft_3 = abs(FFT_SHIFT(dft2D(img_3)))

    img_4 = cv.imread('lunar_surface.tif')
    img_4 = ImgResize(img_4)
    img_4 = ImgChannel(img_4)
    fft_4 = abs(FFT_SHIFT(dft2D(img_4)))

    #结果展示
    plt.title('Characters test')
    plt.imshow(img_1, cmap=plt.get_cmap('gray'))
    plt.show()
    plt.title('Characters test dft2D')
    plt.imshow(np.log(1 + fft_1), cmap=plt.get_cmap('gray'))
    plt.show()

    plt.title('house02')
    plt.imshow(img_2, cmap=plt.get_cmap('gray'))
    plt.show()
    plt.title('house02 dft2D')
    plt.imshow(np.log(1 + fft_2), cmap=plt.get_cmap('gray'))
    plt.show()

    plt.title('rose512')
    plt.imshow(img_3, cmap=plt.get_cmap('gray'))
    plt.show()
    plt.title('rose512 dft2D')
    plt.imshow(np.log(1 + fft_3), cmap=plt.get_cmap('gray'))
    plt.show()

    plt.title('lunar surface')
    plt.imshow(img_4, cmap=plt.get_cmap('gray'))
    plt.show()
    plt.title('lunar surface dft2D')
    plt.imshow(np.log(1 + fft_4), cmap=plt.get_cmap('gray'))
    plt.show()