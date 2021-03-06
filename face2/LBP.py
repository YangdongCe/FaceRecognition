from numpy import *
from numpy import linalg as la
import cv2
import os
import math


# 为了让LBP具有旋转不变性，将二进制串进行旋转。
# 假设一开始得到的LBP特征为10010000，那么将这个二进制特征，
# 按照顺时针方向旋转，可以转化为00001001的形式，这样得到的LBP值是最小的。
# 无论图像怎么旋转，对点提取的二进制特征的最小值是不变的，
# 用最小值作为提取的LBP特征，这样LBP就是旋转不变的了。
def minBinary(pixel):
    length = len(pixel)
    zero = ''
    for i in range(length)[::-1]:
        if pixel[i] == '0':
            pixel = pixel[:i]
            zero += '0'
        else:
            return zero + pixel
    if len(pixel) == 0:
        return '0'

        # 加载图像


def loadImageSet(add):
    FaceMat = mat(zeros((3, 98 * 116)))
    j = 0
    img = cv2.imread('3.jpg', 0)
    FaceMat[j, :] = mat(img).flatten()
    j += 1
    img = cv2.imread(add, 0)
    FaceMat[j, :] = mat(img).flatten()
    j += 1
    img = cv2.imread('4.jpg', 0)
    FaceMat[j, :] = mat(img).flatten()
    j += 1
    # for i in os.listdir(add):
    #     if i.split('.')[1] == 'noglasses':
    #         try:
    #             img = cv2.imread(add  , 0)
    #         except:
    #             print('load %s failed' % i)
    #         FaceMat[j, :] = mat(img).flatten()
    #         j += 1
    return FaceMat


# 算法主过程
def LBP(FaceMat, R=2, P=8):
    Region8_x = [-1, 0, 1, 1, 1, 0, -1, -1]
    Region8_y = [-1, -1, -1, 0, 1, 1, 1, 0]
    pi = math.pi
    LBPoperator = mat(zeros(shape(FaceMat)))
    for i in range(shape(FaceMat)[1]):
        # 对每一个图像进行处理
        face = FaceMat[:, i].reshape(116, 98)
        W, H = shape(face)
        tempface = mat(zeros((W, H)))
        for x in range(R, W - R):
            for y in range(R, H - R):
                repixel = ''
                pixel = int(face[x, y])
                # 　圆形LBP算子
                for p in [2, 1, 0, 7, 6, 5, 4, 3]:
                    p = float(p)
                    xp = x + R * cos(2 * pi * (p / P))
                    yp = y - R * sin(2 * pi * (p / P))
                    if int(face[int(xp), int(yp)]) > pixel:
                        repixel += '1'
                    else:
                        repixel += '0'
                        # minBinary保持LBP算子旋转不变
                tempface[x, y] = int(minBinary(repixel), base=2)
        LBPoperator[:, i] = tempface.flatten().T
        # cv2.imwrite(str(i)+'hh.jpg',array(tempface,uint8))
    return LBPoperator

    # judgeImg:未知判断图像
    # LBPoperator:实验图像的LBP算子
    # exHistograms:实验图像的直方图分布


def judgeFace(judgeImg, LBPoperator, exHistograms):
    judgeImg = judgeImg.T
    ImgLBPope = LBP(judgeImg)
    #  把图片分为7*4份 , calHistogram返回的直方图矩阵有28个小矩阵内的直方图
    judgeHistogram = calHistogram(ImgLBPope)
    minIndex = 10
    minVals = inf

    for i in range(shape(LBPoperator)[1]):
        exHistogram = exHistograms[:, i]
        diff = (array(exHistogram - judgeHistogram) ** 2).sum()
        if diff < minVals:
            minIndex = i
            minVals = diff
    return minIndex


# 统计直方图
def calHistogram(ImgLBPope):
    Img = ImgLBPope.reshape(116, 98)
    W, H = shape(Img)
    # 把图片分为7*4份
    Histogram = mat(zeros((256, 7 * 4)))
    maskx, masky = W / 4, H / 7
    for i in range(4):
        for j in range(7):
            # 使用掩膜opencv来获得子矩阵直方图
            mask = zeros(shape(Img), uint8)
            maskx = int(maskx)
            masky = int(masky)
            mask[i * maskx: (i + 1) * maskx, j * masky:(j + 1) * masky] = 255
            hist = cv2.calcHist([array(Img, uint8)], [0], mask, [256], [0, 256])
            Histogram[:, (i + 1) * (j + 1) - 1] = mat(hist).flatten().T
    return Histogram.flatten().T


def runLBP():
    # 加载图像
    FaceMat = loadImageSet('images.jpg').T

    LBPoperator = LBP(FaceMat)
    # 获得实验图像LBP算子

    # 获得实验图像的直方图分布，这里计算是为了可以多次使用
    exHistograms = mat(zeros((256 * 4 * 7, shape(LBPoperator)[1])))
    for i in range(shape(LBPoperator)[1]):
        exHistogram = calHistogram(LBPoperator[:, i])
        exHistograms[:, i] = exHistogram

        # 　下面的代码都是根据我的这个数据库来的，就是为了验证算法准确性，如果大家改了实例，请更改下面的代码
    nameList = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
    characteristic = ['wu','libinbin', 'fanbinbin', 'aobama', 'normal', 'leftlight', 'noglasses', 'rightlight', 'sad',
                      'sleepy', 'surprised', 'wink']



    loadname = 'shunyang.jpg'
    judgeImg = cv2
    judgeImg = cv2.imread(loadname, 0)
    print(characteristic[judgeFace(mat(judgeImg).flatten(), LBPoperator, exHistograms)])
    loadname = '5.jpg'
    judgeImg = cv2
    judgeImg = cv2.imread(loadname, 0)
    print(characteristic[judgeFace(mat(judgeImg).flatten(), LBPoperator, exHistograms)])



if __name__ == '__main__':
    runLBP()

