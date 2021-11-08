import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

'''
问题1 黑白图像灰度扫描 (20 分)
实现一个函数 s = scanLine4e(f, I, loc), 其中f 是一个灰度图像，I 是一个整数，loc 是一个字
符串。当loc 为’row’时，I 代表行数。当loc 为’column’时，I 代表列数。输出s 是对应的相
关行或者列的像素灰度值矢量。
调用该函数，提取cameraman.tif 和einstein.tif 的中心行和中心列的像素灰度矢量并将扫描
得到的灰度序列绘制成图。
'''


def scanLine4e(f, I, loc):
    s = []  # 灰度值矢量，维度为1
    # 判断位置
    if loc == 'row':
        for j in range(len(f[I, :])):
            s.append(f[I, j])
        return s
    elif loc == 'column':
        for i in range(len(f[:, I])):
            s.append(f[i, I])
        return s
    else:
        raise Exception("Input error,try again!")


if __name__ == '__main__':
    # 导入第一张图片
    im1 = cv.imread('cameraman.tif')
    im1 = np.array(im1)
    h1, w1 = im1.shape[0], im1.shape[1]
    # 调用scanLine4e函数提取中心行和中心列
    s1 = scanLine4e(im1, int(h1 / 2), 'row')
    s2 = scanLine4e(im1, int(w1 / 2), 'column')
    # 导入第二张图片
    im2 = cv.imread('einstein.tif')
    im2 = np.array(im2)
    h2, w2 = im2.shape[0], im2.shape[1]
    s3 = scanLine4e(im2, int(h2 / 2), 'row')
    s4 = scanLine4e(im2, int(w2 / 2), 'column')
    # 成图
    plt.plot(s1)
    plt.title('cameraman center row')
    plt.show()
    plt.plot(s2)
    plt.title('cameraman center column')
    plt.show()
    plt.plot(s3)
    plt.title('einstein center row')
    plt.show()
    plt.plot(s4)
    plt.title('einstein center column')
    plt.show()
