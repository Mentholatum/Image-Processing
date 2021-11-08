from hw2_1 import *

"""
问题4 计算图像的中心化二维快速傅里叶变换与谱图像(12分)
我们的目标是复现下图中的结果。首先合成矩形物体图像，建议图像尺寸为512×512，
矩形位于图像中心，建议尺寸为60像素长，10像素宽，灰度假设已归一化设为1.
对于输入图像f计算其中心化二维傅里叶变换F。然后计算对应的谱图像S=log(1+abs(F)).
显示该谱图像。
"""

def DFT_2(f, M=512, N=512):
    #二维离散傅里叶变换
    if M % 2 != 0 or N % 2 != 0:
        print("Param invalid! Try again.")
        return
    f = f.astype('float64')
    rows = f.shape[0]
    cols = f.shape[1]
    n = np.arange(0, N, 1).reshape((N, 1))
    row = np.arange(0, rows, 1).reshape((1, rows))
    left = np.exp(-1j*2*np.pi/N*(n @ row))
    col = np.arange(0, cols, 1).reshape((cols, 1))
    m = np.arange(0, M, 1).reshape((1, M))
    right = np.exp(-1j*2*np.pi/M*(col @ m))
    F = left @ f @ right
    return F


if __name__ == '__main__':
    #二值图像生成
    img = np.zeros([512,512])
    img[227: 287, 252: 262] = 1
    #获取长宽
    M = np.size(img, 0)
    N = np.size(img, 1)

    #二维离散傅里叶变换
    F_1 = abs(DFT_2(img,M,N))
    #中心化
    F_2 = abs(FFT_SHIFT(DFT_2(img,M,N)))
    #对数变换
    F_3 = np.log(abs(F_2) + 1)
    plt.title('Origin')
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()

    plt.title('Spectrum Image')
    plt.imshow(F_1, cmap=plt.get_cmap('gray'))
    plt.show()

    plt.title('Centralized Spectrum Image')
    plt.imshow(F_2, cmap=plt.get_cmap('gray'))
    plt.show()

    plt.title('Logarithmic Change')
    plt.imshow(F_3, cmap=plt.get_cmap('gray'))
    plt.show()