# -*- coding = utf-8 -*-
# @time:2022/4/24 19:49
# Author:leeJiayi
# @File:ImageDealClass.py
# @Software:PyCharm

from math import *
import numpy as np
import cv2
from scipy import ndimage
from scipy import signal
from PyQt5.QtWidgets import QMessageBox


def frequest(im, orientim, windowsize, minWaveLength, maxWaveLength):
    rows, cols = np.shape(im)
    """
    找到区块内的平均方向。
    这是通过在再次重建角度之前对加倍的角度的正弦和余弦进行平均来实现的。 
    这就避免了原点处的环绕问题。
    """
    cosorient = np.mean(np.cos(2 * orientim))
    sinorient = np.mean(np.sin(2 * orientim))
    orient = atan2(sinorient, cosorient) / 2

    # 旋转图像块，使脊线呈垂直状态
    rotim = ndimage.rotate(im, orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3, mode='nearest')

    # 现在裁剪图像，使旋转后的图像不包含任何无效的区域。 这可以防止列的投影被打乱。
    cropsze = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - cropsze) / 2))
    rotim = rotim[offset:offset + cropsze][:, offset:offset + cropsze]

    # 将各列相加，得到灰度值在脊线上的投影。
    proj = np.sum(rotim, axis=0)
    dilation = ndimage.grey_dilation(proj, windowsize, structure=np.ones(windowsize))

    temp = np.abs(dilation - proj)

    peak_thresh = 2

    maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
    maxind = np.where(maxpts)

    rows_maxind, cols_maxind = np.shape(maxind)

    # 通过将第一和最后一个峰之间的距离除以（峰的数量 - 1）来确定脊线的空间频率。
    # 如果没有检测到峰，或者波长超出了允许的范围，则频率图像被设置为0

    if cols_maxind < 2:
        freqim = np.zeros(im.shape)
    else:
        NoOfPeaks = cols_maxind
        waveLength = (maxind[0][cols_maxind - 1] - maxind[0][0]) / (NoOfPeaks - 1)
        if waveLength >= minWaveLength and waveLength <= maxWaveLength:
            freqim = 1 / np.double(waveLength) * np.ones(im.shape)
        else:
            freqim = np.zeros(im.shape)

    return freqim


class FingerPrintDealing:
    def __init__(self):
        """
        指纹处理类初始化函数
        """

        # 存储原始图像
        self.im = None
        # 存储规格化后的图像
        self.norm_img = None
        # 存储掩膜
        self.mask = None
        # 存储方向图
        self.orientim = None
        # 存储细化后的图像
        self.ThinImg=None
        # 存储图像增强后的图像
        self.enhanceim=None
        # 用于细化图像的数组
        self.ThinArray = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
                          1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                          0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
                          1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                          1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
                          1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                          0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
                          1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                          1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                          1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,
                          1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
        # 存储指纹特征的数组
        self.features=[]

    def GetROIAndSpecification(self, blank_size, threshold):

        """
        ROI区域提取与规格化函数\n

        图像的规格化就是将原始指纹图像的灰度值的均值和方差调整到所期望的均值和方差。\n
        规格化过程并不能够增强脊和谷结构的对比清晰度，主要目的是减少沿脊和谷方向上的灰度级的变化。\n
        :param blank_size: 块大小
        :param threshold: 阈值
        :return:
        """

        if self.im.any():
            rows, cols = self.im.shape
            im = (self.im - np.mean(self.im)) / (np.std(self.im))

            new_rows = np.int(blank_size * np.ceil((np.float(rows)) / (np.float(blank_size))))
            new_cols = np.int(blank_size * np.ceil((np.float(cols)) / (np.float(blank_size))))
            padded_img = np.zeros((new_rows, new_cols))
            stddev_im = np.zeros((new_rows, new_cols))

            padded_img[0:rows][:, 0:cols] = im

            for i in range(0, new_rows, blank_size):
                for j in range(0, new_cols, blank_size):
                    # 将指纹图像划分成块
                    block = padded_img[i:i + blank_size][:, j:j + blank_size]

                    stddev_im[i:i + blank_size][:, j:j + blank_size] = np.std(block) * np.ones(block.shape)

            stddev_im = stddev_im[0:rows][:, 0:cols]
            # 创建指纹图像掩膜
            self.mask = stddev_im > threshold
            mean_val = np.mean(im[self.mask])
            std_val = np.std(im[self.mask])
            # 将图像规格化
            self.norm_img = (im - mean_val) / (std_val)
        else:
            e = QMessageBox(QMessageBox.Information, "错误", "无法创建掩膜与规格化图像，请初始化图像处理类的原始图像！")
            e.exec_()

    def get_orient(self, gradientsigma, blocksigma, orientsmoothsigma):
        """
        获取指纹的方向图
        :param gradientsigma: 用于平滑梯度的高斯函数Sigma值
        :param blocksigma: 用于平滑协方差的高斯函数Sigma值
        :param orientsmoothsigma: 用于平滑方向图的高斯函数Sigma值
        :return:
        """
        if self.norm_img.all():
            # 计算图像梯度
            sze = np.fix(6 * gradientsigma)
            if np.remainder(sze, 2) == 0:
                sze = sze + 1

            gauss = cv2.getGaussianKernel(np.int(sze), gradientsigma)
            f = gauss * gauss.T # 7*7

            fy, fx = np.gradient(f)  # 高斯梯度

            # Gx = ndimage.convolve(np.double(im),fx);
            # Gy = ndimage.convolve(np.double(im),fy);

            Gx = signal.convolve2d(self.norm_img, fx, mode='same')
            Gy = signal.convolve2d(self.norm_img, fy, mode='same')

            Gxx = np.power(Gx, 2)
            Gyy = np.power(Gy, 2)
            Gxy = Gx * Gy

            # 对协方差数据进行平滑处理，对数据进行加权求和

            sze = np.fix(6 * blocksigma)

            gauss = cv2.getGaussianKernel(np.int(sze), blocksigma)
            f = gauss * gauss.T

            Gxx = ndimage.convolve(Gxx, f)
            Gyy = ndimage.convolve(Gyy, f)
            Gxy = 2 * ndimage.convolve(Gxy, f)

            # 主方向的解析解
            denom = np.sqrt(np.power(Gxy, 2) + np.power((Gxx - Gyy), 2)) + np.finfo(float).eps

            sin2theta = Gxy / denom  # 二倍角的正弦和余弦
            cos2theta = (Gxx - Gyy) / denom

            if orientsmoothsigma:
                sze = np.fix(6 * orientsmoothsigma)
                if np.remainder(sze, 2) == 0:
                    sze = sze + 1
                gauss = cv2.getGaussianKernel(np.int(sze), orientsmoothsigma)
                f = gauss * gauss.T
                # 平滑处理后二倍角的正弦和余弦
                cos2theta = ndimage.convolve(cos2theta, f)
                sin2theta = ndimage.convolve(sin2theta, f)

            self.orientim = np.pi / 2 + np.arctan2(sin2theta, cos2theta) / 2

        else:
            e = QMessageBox(QMessageBox.Information, "错误", "无法获得方向图，请先获取图像处理类的规格化图像！")
            e.exec_()

    def get_frequest(self, blksze, windsze, minWaveLength, maxWaveLength):
        '''
        获取指纹图像的频率阵列，为gabor滤波做准备\n
        :param blksze: 图像块大小
        :param windsze: 方向窗大小
        :param minWaveLength: 最小波长
        :param maxWaveLength: 最大波长
        :return:
        '''
        if self.norm_img.all() and self.orientim.all():
            rows, cols = self.norm_img.shape
            freq = np.zeros((rows, cols))

            for r in range(0, rows - blksze, blksze):
                for c in range(0, cols - blksze, blksze):
                    blkim = self.norm_img[r:r + blksze][:, c:c + blksze]
                    blkor = self.orientim[r:r + blksze][:, c:c + blksze]

                    freq[r:r + blksze][:, c:c + blksze] = frequest(blkim, blkor, windsze, minWaveLength, maxWaveLength)

            freq = freq * self.mask
            freq_1d = np.reshape(freq, (1, rows * cols))
            ind = np.where(freq_1d > 0)

            ind = np.array(ind)
            ind = ind[1, :]

            non_zero_elems_in_freq = freq_1d[0][ind]

            self.meanfreq = np.mean(non_zero_elems_in_freq)
            self.freq = self.meanfreq * self.mask

        else:
            e = QMessageBox(QMessageBox.Information, "错误", "无法获得频率场，请获取图像处理类的分割图像与方向图！")
            e.exec_()

    def gabor_filter(self, kx, ky):
        '''
        Gabor滤波法指纹图像增强：是使指纹脊线结构更清晰，保留和突出指纹固有信息。\n
        :param kx: 高斯包络沿x轴的空间常数
        :param ky: 高斯包络沿y轴的空间常数
        :return:
        '''
        if self.norm_img.all() and self.orientim.all():
            angleInc = 3
            im = np.double(self.norm_img)
            rows, cols = im.shape
            newim = np.zeros((rows, cols))

            freq_1d = np.reshape(self.freq, (1, rows * cols))
            ind = np.where(freq_1d > 0)

            ind = np.array(ind)
            ind = ind[1, :]

            # 将频率阵列四舍五入到最接近的0.01，以减少我们必须处理的不同频率的数量。

            non_zero_elems_in_freq = freq_1d[0][ind]
            non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100

            unfreq = np.unique(non_zero_elems_in_freq)

            # 以'angleInc'的增量生成对应于这些不同频率和方向的滤波器。

            sigmax = 1 / unfreq[0] * kx
            sigmay = 1 / unfreq[0] * ky

            sze = np.int(np.round(3 * np.max([sigmax, sigmay])))

            x, y = np.meshgrid(np.linspace(-sze, sze, (2 * sze + 1)), np.linspace(-sze, sze, (2 * sze + 1)))

            reffilter = np.exp(
                -(((np.power(x, 2)) / (sigmax * sigmax) + (np.power(y, 2)) / (sigmay * sigmay)))) * np.cos(
                2 * np.pi * unfreq[0] * x)  # 这就是原始的Gabor滤波器

            filt_rows, filt_cols = reffilter.shape

            angleRange = np.int(180 / angleInc)

            gabor_filter = np.array(np.zeros((angleRange, filt_rows, filt_cols)))

            for o in range(0, angleRange):
                #生成过滤器的旋转版本。注意方向图像提供的方向是*沿着脊线的，因此是+90度，而im旋转需要逆时针+ve的角度，因此是减号。

                rot_filt = ndimage.rotate(reffilter, -(o * angleInc + 90), reshape=False)
                gabor_filter[o] = rot_filt

            # 找到距离图像边界大于maxsze的矩阵点的索引
            maxsze = int(sze)

            temp = self.freq > 0
            validr, validc = np.where(temp)

            temp1 = validr > maxsze
            temp2 = validr < rows - maxsze
            temp3 = validc > maxsze
            temp4 = validc < cols - maxsze

            final_temp = temp1 & temp2 & temp3 & temp4

            finalind = np.where(final_temp)

            # 将方向矩阵值从弧度转换为与round(degrees/angleInc)相对应的索引值。

            maxorientindex = np.round(180 / angleInc)
            orientindex = np.round(self.orientim / np.pi * 180 / angleInc)

            # 进行滤波

            for i in range(0, rows):
                for j in range(0, cols):
                    if orientindex[i][j] < 1:
                        orientindex[i][j] = orientindex[i][j] + maxorientindex
                    if orientindex[i][j] > maxorientindex:
                        orientindex[i][j] = orientindex[i][j] - maxorientindex
            finalind_rows, finalind_cols = np.shape(finalind)
            sze = int(sze)
            for k in range(0, finalind_cols):
                r = validr[finalind[0][k]]
                c = validc[finalind[0][k]]

                img_block = im[r - sze:r + sze + 1][:, c - sze:c + sze + 1]

                newim[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r][c]) - 1])

            self.newim = newim
            self.enhanceim = 255 * (newim >= -3)

        else:
            e = QMessageBox(QMessageBox.Information, "错误", "无法进行gabor滤波，请获取图像处理类的规格化图像与方向图！")
            e.exec_()

    def image_enhance(self):
        """
        图像增强总调用函数

        :return: None
        """
        self.GetROIAndSpecification(blank_size=16, threshold=0.1)
        # cv2.imshow("norm", self.norm_img)
        self.get_orient(gradientsigma=1, blocksigma=7, orientsmoothsigma=7)
        # cv2.imshow("orient", self.orientim)
        self.get_frequest(blksze=16, windsze=5, minWaveLength=5, maxWaveLength=15)
        self.gabor_filter(kx=0.65, ky=0.65)
        # cv2.imshow("new", self.newim)
        # cv2.imshow("new", self.enhanceim)

    def Thin(self, image, num=10):
        """
        主要采用的是索引表细化方法\n
        :param image: 需要细化的图像
        :param num: 迭代次数
        :return:
        """
        try:
            h, w = image.shape
            for nu in range(num):
                NEXT = 1
                """
                水平扫描
                """
                for i in range(h):
                    for j in range(w):
                        if NEXT == 0:
                            NEXT = 1
                        else:
                            M = image[i, j - 1] + image[i, j] + image[i, j + 1] if 0 < j < w - 1 else 1
                            if image[i, j] == 0 and M != 0:
                                a = [0] * 9
                                for k in range(3):
                                    for l in range(3):
                                        if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 255:
                                            a[k * 3 + l] = 1
                                sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[
                                    8] * 128
                                image[i, j] = self.ThinArray[sum] * 255
                                if self.ThinArray[sum] == 1:
                                    NEXT = 0
                """
                竖直扫描
                """
                NEXT = 1
                for j in range(w):
                    for i in range(h):
                        if NEXT == 0:
                            NEXT = 1
                        else:
                            M = image[i - 1, j] + image[i, j] + image[i + 1, j] if 0 < i < h - 1 else 1
                            if image[i, j] == 0 and M != 0:
                                a = [0] * 9
                                for k in range(3):
                                    for l in range(3):
                                        if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 255:
                                            a[k * 3 + l] = 1
                                sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[
                                    8] * 128
                                image[i, j] = self.ThinArray[sum] * 255
                                if self.ThinArray[sum] == 1:
                                    NEXT = 0

            self.ThinImg=image
        except Exception as e:
            e = QMessageBox(QMessageBox.Information, "错误", str(e))
            e.exec_()

    def getFeature(self,img):
        """
        获取指纹图像特征\n
        :param img: 预处理后的指纹图像
        :return: None
        """
        try:
            h, w = img.shape
            for i in range(1,h - 1):
                for j in range(1, w - 1):
                    if img[i, j] == 0:# 对像素点为黑的进行测试
                        x = i
                        y = j
                        eightNeighborhood=[img[x - 1, y - 1], img[x - 1, y], img[x - 1, y + 1], img[x, y - 1], img[x, y + 1],
                                           img[x + 1, y - 1], img[x + 1, y], img[x + 1, y + 1]]

                        if sum(eightNeighborhood) / 255 == 7:  # 黑色块1个，端点

                            # 判断是否为指纹图像边缘
                            if sum(img[:i, j]) == 255 * i or sum(img[i + 1:, j]) == 255 * (w - i - 1) or sum(
                                    img[i, :j]) == 255 * j or sum(img[i, j + 1:]) == 255 * (h - j - 1):
                                continue
                            canContinue = True
                            # print(x, y)
                            coordinate = [[x - 1, y - 1], [x - 1, y], [x - 1, y + 1], [x, y - 1], [x, y + 1],
                                          [x + 1, y - 1],
                                          [x + 1, y], [x + 1, y + 1]]
                            for o in range(8):  # 寻找相连接的下一个点
                                if eightNeighborhood[o] == 0:
                                    index = o
                                    x = coordinate[o][0]
                                    y = coordinate[o][1]
                                    # print(x, y, index)
                                    break
                            # print(x, y, index)
                            for k in range(4):
                                coordinate = [[x - 1, y - 1], [x - 1, y], [x - 1, y + 1], [x, y - 1], [x, y + 1],
                                              [x + 1, y - 1], [x + 1, y], [x + 1, y + 1]]
                                eightNeighborhood = [img[x - 1, y - 1], img[x - 1, y], img[x - 1, y + 1], img[x, y - 1],
                                                     img[x, y + 1],
                                                     img[x + 1, y - 1], img[x + 1, y], img[x + 1, y + 1]]
                                if sum(eightNeighborhood) / 255 == 6:  # 连接点
                                    for o in range(8):
                                        if eightNeighborhood[o] == 0 and o != 7 - index:
                                            index = o
                                            x = coordinate[o][0]
                                            y = coordinate[o][1]
                                            # print(x, y, index)
                                            break
                                else:
                                    # print("false", i, j)
                                    canContinue = False
                            if canContinue:

                                if y - j != 0:
                                    if i - x >= 0 and j - y > 0:
                                        angle = atan((i - x) / (y - j)) + pi
                                    elif i - x < 0 and j - y > 0:
                                        angle = atan((i - x) / (y - j)) - pi
                                    else:
                                        angle = atan((i - x) / (y - j))
                                else:
                                    if i - x >= 0:
                                        angle = pi / 2
                                    else:
                                        angle = -pi / 2
                                feature = []

                                angle=degrees(angle)
                                feature.append(j)
                                feature.append(i)
                                feature.append("端点")
                                feature.append(angle)
                                self.features.append(feature)

                        elif sum(eightNeighborhood) / 255 == 5:  # 黑色块3个，分叉点
                            coordinate = [[x - 1, y - 1], [x - 1, y], [x - 1, y + 1], [x, y - 1], [x, y + 1],
                                          [x + 1, y - 1],
                                          [x + 1, y], [x + 1, y + 1]]
                            junctionCoordinates = []
                            junctions = []
                            canContinue = True
                            # 筛除不符合的分叉点
                            for o in range(8):  # 寻找相连接的下一个点
                                if eightNeighborhood[o] == 0:
                                    junctions.append(o)
                                    junctionCoordinates.append(coordinate[o])
                            for k in range(3):
                                if k == 0:
                                    a = junctions[0]
                                    b = junctions[1]
                                elif k == 1:
                                    a = junctions[1]
                                    b = junctions[2]
                                else:
                                    a = junctions[0]
                                    b = junctions[2]
                                if (a == 0 and b == 1) or (a == 1 and b == 2) or (a == 2 and b == 4) or (
                                        a == 4 and b == 7) or (
                                        a == 6 and b == 7) or (a == 5 and b == 6) or (a == 3 and b == 5) or (
                                        a == 0 and b == 3):
                                    canContinue = False
                                    break

                            if canContinue:  # 合格分叉点
                                # print(junctions)
                                # print(junctionCoordinates)
                                # print(i, j, "合格分叉点")
                                directions = []
                                canContinue = True
                                for k in range(3):  # 分三路进行
                                    if canContinue:
                                        junctionCoordinate = junctionCoordinates[k]
                                        x = junctionCoordinate[0]
                                        y = junctionCoordinate[1]
                                        # print(x, y, "start")
                                        eightNeighborhood = [img[x - 1, y - 1], img[x - 1, y], img[x - 1, y + 1], img[x, y - 1],
                                                             img[x, y + 1],
                                                             img[x + 1, y - 1], img[x + 1, y], img[x + 1, y + 1]]
                                        coordinate = [[x - 1, y - 1], [x - 1, y], [x - 1, y + 1], [x, y - 1], [x, y + 1],
                                                      [x + 1, y - 1], [x + 1, y], [x + 1, y + 1]]
                                        canContinue = False
                                        for o in range(8):
                                            if eightNeighborhood[o] == 0:
                                                a = coordinate[o][0]
                                                b = coordinate[o][1]
                                                #  print("a=", a, "b=", b)
                                                # print("i=", i, "j=", j)
                                                if (a != i or b != j) and (
                                                        a != junctionCoordinates[0][0] or b != junctionCoordinates[0][
                                                    1]) and (
                                                        a != junctionCoordinates[1][0] or b != junctionCoordinates[1][
                                                    1]) and (
                                                        a != junctionCoordinates[2][0] or b != junctionCoordinates[2][1]):
                                                    index = o
                                                    x = a
                                                    y = b
                                                    canContinue = True
                                                    # print(x, y, index, "支路", k)
                                                    break
                                        if canContinue:  # 能够找到第二个支路点
                                            for p in range(3):
                                                coordinate = [[x - 1, y - 1], [x - 1, y], [x - 1, y + 1], [x, y - 1],
                                                              [x, y + 1],
                                                              [x + 1, y - 1], [x + 1, y], [x + 1, y + 1]]
                                                eightNeighborhood = [img[x - 1, y - 1], img[x - 1, y], img[x - 1, y + 1],
                                                                     img[x, y - 1],
                                                                     img[x, y + 1],
                                                                     img[x + 1, y - 1], img[x + 1, y], img[x + 1, y + 1]]
                                                if sum(eightNeighborhood) / 255 == 6:  # 连接点
                                                    for o in range(8):
                                                        if eightNeighborhood[o] == 0 and o != 7 - index:
                                                            index = o
                                                            x = coordinate[o][0]
                                                            y = coordinate[o][1]
                                                            # print(x, y, index, "支路尾")
                                                            # print(x, y, index)
                                                            break
                                                else:
                                                    # print("false", i, j)
                                                    canContinue = False
                                        if canContinue:  # 能够找到3个连接点

                                            if y - j != 0:
                                                if i - x >= 0 and j - y > 0:
                                                    angle = atan((i - x) / (y - j)) + pi
                                                elif i - x < 0 and j - y > 0:
                                                    angle = atan((i - x) / (y - j)) - pi
                                                else:
                                                    angle = atan((i - x) / (y - j))
                                            else:
                                                if i - x >= 0:
                                                    angle = pi / 2
                                                else:
                                                    angle = -pi / 2
                                            # print(direction)
                                            angle=degrees(angle)
                                            directions.append(angle)
                                if canContinue:
                                    feature = []
                                    feature.append(j)
                                    feature.append(i)
                                    feature.append("三叉点")
                                    feature.append(directions)
                                    self.features.append(feature)
            self.featureImg=img
            for m in range(len(self.features)):
                if self.features[m][2] == "端点":
                    cv2.circle(self.featureImg, (self.features[m][0], self.features[m][1]), 3, (0, 0, 255), 1,cv2.FILLED)
                else:
                    cv2.circle(self.featureImg, (self.features[m][0], self.features[m][1]), 3, (0, 255, 0), -1,cv2.FILLED)
        except Exception as e:
            e = QMessageBox(QMessageBox.Information, "错误", "无法进行特征提取\n错误提示如下："+str(e))
            e.exec_()