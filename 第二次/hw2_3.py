from hw2_2 import *

'''
问题3 测试图像二维快速傅里叶变换与逆变换（8分）
对于给定的输入图像rose512.tif, 首先将其灰度范围通过归一化调整到[0,1]. 
将此归一化的图像记为f. 首先调用问题1下实现的函数dft2D计算其傅里叶变换，记为F。
然后调用问题2下的函数idft2D计算F的傅里叶逆变换，记为g. 计算并显示误差图像d = f-g.
'''

if __name__ == '__main__':
    img = cv.imread('rose512.tif',cv.IMREAD_GRAYSCALE)
    #归一化
    f = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    M = np.size(img,0)
    N = np.size(img,1)

    #计算结果
    f = f / (M * N)
    F = dft2D(f)
    g = idft2D(F)
    d = f - g

    #显示结果
    plt.title('Rose')
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()
    plt.title('Rose after fourier transform')
    img_n = abs(g)
    plt.imshow(img_n, cmap=plt.get_cmap('gray'))
    plt.show()
    print("Deviation:",d)





