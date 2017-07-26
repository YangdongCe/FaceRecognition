from numpy import *
from numpy import linalg as la
import cv2
import os

characteristic = ['wu' ,'libinbin', 'fanbinbin', 'aobama','shunyang','wu']
def loadImageSet(add):
    FaceMat = mat(zeros((5, 98 * 116)))
    j = 0
    img = cv2.imread('3.jpg', 0)
    FaceMat[j, :] = mat(img).flatten()
    j += 1
    img = cv2.imread('2.jpg', 0)
    FaceMat[j, :] = mat(img).flatten()
    j += 1
    img = cv2.imread('5.jpg', 0)
    FaceMat[j, :] = mat(img).flatten()
    j += 1
    img = cv2.imread('6.jpg', 0)
    FaceMat[j, :] = mat(img).flatten()
    j += 1
    img = cv2.imread(add, 0)
    FaceMat[j, :] = mat(img).flatten()
    return FaceMat


def ReconginitionVector(imgname ,selecthr=0.8):
    # step1: load the face image data ,get the matrix consists of all image
    FaceMat = loadImageSet(imgname).T
    # step2: average the FaceMat
    avgImg = mean(FaceMat, 1)
    # step3: calculate the difference of avgimg and all image data(FaceMat)
    diffTrain = FaceMat - avgImg
    # step4: calculate eigenvector of covariance matrix (because covariance matrix will cause memory error)
    eigvals, eigVects = linalg.eig(mat(diffTrain.T * diffTrain))
    eigSortIndex = argsort(-eigvals)
    for i in range(shape(FaceMat)[1]):
        if (eigvals[eigSortIndex[:i]] / eigvals.sum()).sum() >= selecthr:
            eigSortIndex = eigSortIndex[:i]
            break
    covVects = diffTrain * eigVects[:, eigSortIndex]  # covVects is the eigenvector of covariance matrix
    # avgImg 是均值图像，covVects是协方差矩阵的特征向量，diffTrain是偏差矩阵
    return avgImg, covVects, diffTrain


def judgeFace(judgeImg, FaceVector, avgImg, diffTrain):
    diff = judgeImg.T - avgImg
    weiVec = FaceVector.T * diff
    res = 0
    resVal = inf
    for i in range(4):
        TrainVec = FaceVector.T * diffTrain[:, i]
        print(res)
        if (array(weiVec - TrainVec) ** 2).sum() < resVal:
            res = i
            resVal = (array(weiVec - TrainVec) ** 2).sum()

    return res+1


if __name__ == '__main__':
    loadname = 'images.jpg'
    avgImg, FaceVector, diffTrain = ReconginitionVector(loadname ,selecthr=0.8)
    judgeImg = cv2.imread(loadname, 0)
    print(characteristic[judgeFace(mat(judgeImg).flatten(), FaceVector, avgImg, diffTrain)])
    # for c in characteristic:
    #
    #     count = 0
    #     for i in range(len(nameList)):
    #
    #         # 这里的loadname就是我们要识别的未知人脸图，我们通过15张未知人脸找出的对应训练人脸进行对比来求出正确率
    #         loadname = '2.jpg'
    #         judgeImg = cv2.imread(loadname, 0)
    #         # if judgeFace(mat(judgeImg).flatten(), FaceVector, avgImg, diffTrain) == int(nameList[i]):
    #         #     count += 1
    #     print('accuracy of %s is %f' % (c, float(count) / len(nameList)))  # 求出正确率