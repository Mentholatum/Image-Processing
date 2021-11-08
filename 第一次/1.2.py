import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

'''
问题2 彩色图像转换为黑白图像(20 分)
图像处理中的一个常见问题是将彩色RGB 图像转换成单色灰度图像，第一种常用的方法是
取三个元素R，G，B 的均值。第二种常用的方式，又称为NTSC 标准，考虑了人类的彩色
感知体验，对于R,G,B 三通道分别采用了不同的加权系数，分别是R 通道0.2989，G 通道
0.5870，B 通道0.1140. 实现一个函数 g = rgb1gray(f, method). 函数功能是将一幅 24 位的
RGB 图像, f, 转换成灰度图像, g. 参数 method 是一个字符串，当其值为’average’ 时，采用
第一种转换方法，当其值为’NTSC’时，采用第二种转换方法。将’NTSC’做为缺省方式。
调用该函数，将提供的图像mandril_color.tif 和lena512color.tiff 用上述两种方法转换成单色
灰度图像，对于两种方法的结果进行简短比较和讨论。
'''


def rgb1gray(f, method='NTSC'):
    # 灰度转换函数 NTSC为缺省方式
    if method == 'average':  # 取三元素RGB均值
        g = (f[:, :, 0] + f[:, :, 1] + f[:, :, 2]) / 3
        g = g.clip(0, 1)
        return g
    elif method == 'NTSC':  # NTSC 标准
        g = 0.2989 * f[:, :, 0] + 0.5870 * f[:, :, 1] + 0.1140 * f[:, :, 2]
        g = g.clip(0, 1)
        return g
    else:
        raise Exception("Input error,try again!")


def normalization(data):
    # 归一化函数:为了数据处理方便，把数据映射到0～1范围之内处理，提高数据表现。
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


if __name__ == '__main__':
    # 图片导入
    img1 = cv.imread('mandril_color.tif')
    img2 = cv.imread('lena512color.tiff')
    # 归一化
    img1 = normalization(img1)
    img2 = normalization(img2)
    # NTSC方式转灰度图
    g1 = rgb1gray(img1)
    g2 = rgb1gray(img2)
    # average方式转灰度图
    g3 = rgb1gray(img1, 'average')
    g4 = rgb1gray(img2, 'average')
    # 对比作差
    diff1 = np.abs(g1 - g3)
    diff2 = np.abs(g2 - g4)

    # mandril_color结果展示
    cv.imshow('mandril_color origin', img1)
    cv.imshow('mandril_color NTSC', g1)
    cv.imshow('mandril_color average', g3)
    cv.imshow('mandril_color diff', diff1)

    # lena512color结果展示
    cv.imshow('lena512color origin', img2)
    cv.imshow('lena512color NTSC', g2)
    cv.imshow('lena512color average', g4)
    cv.imshow('lena512color diff', diff2)

    cv.waitKey(0)
    cv.destroyAllWindows()
