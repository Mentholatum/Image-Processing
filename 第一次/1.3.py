import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

'''
图像二维卷积函数 (20 分)
实现一个函数 g = twodConv(f, w), 其中f 是一个灰度源图像，w 是一个矩形卷积核。要求输
出图像g 与源图像f 大小（也就是像素的行数和列数）一致。请注意，为满足这一要求，对
于源图像f 需要进行边界像素填补(padding)。这里请实现两种方案。第一种方案是像素复制，
对应的选项定义为’replicate’，填补的像素拷贝与其最近的图像边界像素灰度。第二种方案是
补零，对应的选项定义为’zero’, 填补的像素灰度为0. 将第二种方案设置为缺省选择。
'''


def twodConv(f, w, method='zero'):
    # 卷积函数，补0方案为缺省选择
    w = np.array(w)
    # 滤波核反转
    w = np.fliplr(np.flipud(w))
    x, y = w.shape
    fh, fw = f.shape
    nh = fh + x - 1
    nw = fw + y - 1
    add_h = int(x) // 2
    add_w = int(y) // 2

    # 零填充边界
    n = np.zeros((nh, nw))
    g = np.zeros((fh, fw))
    # 复制原图
    n[add_h:nh - add_h, add_w:nw - add_w] = f
    if method == 'replicate':
        # 边界填充，填补的像素拷贝与其最近的图像边界像素灰度
        n[0:add_h, add_w:nw - add_w] = f[0, :]
        n[nh - add_h:, add_w:nw - add_w] = f[-1, :]
        for i in range(add_w):
            n[:, i] = n[:, add_w]
            n[:, nw - 1 - i] = n[:, nw - 1 - add_w]
        # 卷积运算
        for i in range(fh):
            for j in range(fw):
                g[i, j] = np.sum(n[i:i + x, j:j + y] * w)
        g = g.clip(0, 255)
        return g
    if method == 'zero':
        for i in range(fh):
            for j in range(fw):
                g[i, j] = np.sum(n[i:i + x, j:j + y] * w)
        g = g.clip(0, 255)
        return g
    else:
        raise Exception("Input error,try again!")


if __name__ == '__main__':
    f = cv.imread("einstein.tif", cv.IMREAD_GRAYSCALE)
    w = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    g1 = twodConv(f, w, method='replicate')
    g2 = twodConv(f, w)#0填充方案
    g3 = np.abs(g1 - g2)
    cv.imshow('einstein origin', f)
    cv.imshow('einstein replicate', g1)
    cv.imshow('einstein zero', g2)
    cv.imshow('einstein diff', g3)

    cv.waitKey(0)
    cv.destroyAllWindows()
