import warnings
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

'''
灰度图像的高斯滤波 (20 分)
调用上面实现的函数，对于问题1 和2 中的灰度图像（cameraman, einstein, 以及lena512color
和mandril_color 对应的NTSC 转换后的灰度图像）进行高斯滤波，采用σ=1，2，3，5。任
选一种像素填补方案。
对于σ=1 下的结果，与直接调用MATLAB 或者Python 软件包相关函数的结果进行比较（可
以简单计算差值图像）。然后，任选两幅图像，比较其他参数条件不变的情况下像素复制和
补零下滤波结果在边界上的差别。
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

def gaussKernel(sig,m=0):
    if sig == None:
        raise Exception('sig invalid,try again!')
    else:
        if m == 0:#mweibo给出时，计算m大小
            m = 1 + 2 * int(3 * sig)
            mCenter = m // 2
            # 创建m阶方阵w
            w = np.zeros((m, m), dtype=np.float64)
            # 高斯核生成
            for x in range(-mCenter, -mCenter + m):
                for y in range(-mCenter, -mCenter + m):
                    w[y + mCenter, x + mCenter] = np.exp(-(x ** 2 + y ** 2) / (2 * (sig ** 2)))
            # 归一化
            w /= (sig * np.sqrt(2 * np.pi))
            w /= w.sum()
            return w
        elif m < 2:
            warnings.warn("m is too small,try again!")
        else:
            mCenter =m // 2
            #创建m阶方阵w
            w = np.zeros((m,m), dtype = np.float64)
            #高斯核生成
            for x in range(-mCenter, -mCenter + m):
                for y in range(-mCenter, -mCenter + m):
                    w[y + mCenter, x + mCenter] = np.exp(-(x ** 2 + y ** 2) / (2 * (sig ** 2)))
            #归一化
            w /= (sig * np.sqrt(2 * np.pi))
            w /= w.sum()
            return w

def normalization(data):
    # 归一化函数:为了数据处理方便，把数据映射到0～1范围之内处理，提高数据表现。
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def twodConv(f, w, method='zero'):
    # 卷积函数，补0方案为缺省选择
    w = np.array(w)
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
    sig = [1, 2, 3, 5]
    # 导入图片
    f1= cv.imread("cameraman.tif",cv.IMREAD_GRAYSCALE)
    f2 = cv.imread("einstein.tif",cv.IMREAD_GRAYSCALE)
    f3 = cv.imread("mandril_color.tif")
    f4 = cv.imread("lena512color.tiff")
    #彩色图转灰度图
    f3 = rgb1gray(normalization(f3))
    f4 = rgb1gray(normalization(f4))

    #对于不同的sigma大小，输出图像
    for i in sig:
        w = gaussKernel(i,0)
        g1 = twodConv(normalization(f1), w)
        g2 = twodConv(normalization(f2), w)
        g3 = twodConv(f3, w)
        g4 = twodConv(f4, w)
        cv.imshow('When sig = '+str(i)+',cameraman', g1)
        cv.imshow('When sig = '+str(i)+',einstein', g2)
        cv.imshow('When sig = '+str(i)+',mandril gray', g3)
        cv.imshow('When sig = '+str(i)+',lena512 gray', g4)

    #sig=1时，与直接调用相关函数的结果进行比较
    w = gaussKernel(1,0)
    g11 = twodConv(normalization(f1), w)
    g12 = twodConv(normalization(f2), w)
    g13 = twodConv(f3, w)
    g14 = twodConv(f4, w)
    F = [f1,f2,f3,f4]
    G = [g11,g12,g13,g14]
    OG = []
    C = []
    #直接调用opencv库函数
    for i in F:
        og = cv.GaussianBlur(i,(7,7),1,borderType=cv.BORDER_REPLICATE)
        og =  normalization(og)
        OG.append(og)
    OG = np.array(OG)
    G = np.array(G)
    C = np.abs(OG-G)
    for i in range(4):
        cv.imshow('Img ' + str(i) +' Use OpenCV', OG[i])
        cv.imshow('Img ' + str(i) + ',difference', C[i])


    #对比像素复制和补零下滤波结果在边界上的差别
    for i in sig:
        w = gaussKernel(i,0)
        g1 = twodConv(normalization(f1), w, method='replicate')
        g2 = twodConv(normalization(f2), w, method='replicate')
        g3 = twodConv(normalization(f1), w)
        g4 = twodConv(normalization(f2), w)
        cv.imshow('When sig = ' + str(i) + ' ,cameraman difference', np.abs(g3-g1))
        cv.imshow('When sig = ' + str(i) + ' ,einstein difference', np.abs(g4-g2))

    cv.waitKey(0)
    cv.destroyAllWindows()

