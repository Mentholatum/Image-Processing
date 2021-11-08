import warnings
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


'''归一化二维高斯滤波核函数 (20 分)
实现一个高斯滤波核函数 w = gaussKernel(sig，m)，其中sig 对应于高斯函数定义中的σ,w
的大小为m×m。请注意，这里如果m 没有提供，需要进行计算确定。如果m 已提供但过小，
应给出警告信息提示。w 要求归一化，即全部元素加起来和为1。'''


def gaussKernel(sig,m=0):
    if sig == None:
        raise Exception('sig invalid,try again!')
    else:
        if m == 0:#m未给出时，计算m大小
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

if __name__ == '__main__':
    sig = float(input("Please input the parameter sigma:"))
    m = int(input("Input the size of guass kernel(When input 0,calculate automatically):"))
    print(gaussKernel(sig,m))
