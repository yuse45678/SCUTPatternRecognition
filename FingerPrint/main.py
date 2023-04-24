# -*- coding = utf-8 -*-
# @time:2022/4/24 10:50
# Author:leeJiayi
# @File:main.py
# @Software:PyCharm

import os
import sys

import cv2
import cv2.cv2
import numpy as np

from PyQt5.QtGui import QImage, QPixmap, QPalette, QBrush
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtWidgets import QMessageBox

import GUI
from ImageDealClass import FingerPrintDealing

# python -m PyQt5.uic.pyuic GUI.ui -o GUI.py转换UI文件！
# pipreqs ./ --encoding=utf8 导出需求环境
# auto-py-to-exe 生成exe'文件

np.set_printoptions(threshold=np.inf, linewidth=850)


class FingerPrintRecongnitionSystem(QMainWindow):

    def __init__(self):
        self.app = QApplication(sys.argv)
        super().__init__()
        self.ui = GUI.Ui_MainWindow()
        self.ui.setupUi(self)
        self.background = QPixmap('background.png')
        # 初始化
        self.init_ui()
        self.img = None
        # 初始化指纹图像处理类
        self.ImageDealClass = FingerPrintDealing()

    # ui初始化
    def init_ui(self):
        # 初始化方法，这里可以写按钮绑定等的一些初始函数
        self.ui.centralwidget.setFixedSize(820, 940)
        self.ui.BeginTestButton.setEnabled(0)
        self.ui.StepByStepBox.setEnabled(0)
        self.ui.SaveButton.setEnabled(0)
        # 选择图像按钮
        self.ui.SelectImageButton.clicked.connect(self.SelectImage)
        # 增强图像按钮
        self.ui.EnhanceButton.clicked.connect(self.EnhanceImage)
        # 细化图像按钮
        self.ui.ThinButton.clicked.connect(self.ThinImage)
        # 提取特征按钮
        self.ui.FeatureButton.clicked.connect(self.GetFeature)
        # 开始提取按钮
        self.ui.BeginTestButton.clicked.connect(self.OneKeyTest)
        # 保存结果按钮
        self.ui.SaveButton.clicked.connect(self.SaveFeature)
        # 存储图像增强图像中间文件的文件夹
        dirs = 'temp'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        # 设置GUI背景
        palette1 = QPalette()
        palette1.setBrush(self.backgroundRole(), QBrush(self.background))  # 背景图片
        self.setPalette(palette1)
        self.setAutoFillBackground(True)
        self.show()

    def clearAll(self):
        # 清空所有输出
        self.ui.FilenameBrowser.setText("")
        self.ui.ImageView.clear()
        self.ui.EnhanceView.clear()
        self.ui.ThinView.clear()
        self.ui.FeatureView.clear()
        self.ui.ResTextBrowser.clear()
        self.ImageDealClass.features = []

    def SelectImage(self):
        """
        选择指纹图片并加载
        """
        try:
            self.clearAll()
            file_path, _ = QFileDialog.getOpenFileName(self, "选择指纹图像文件", os.getcwd(),
                                                       "tif(*.tif);;jpg(*.jpg);;png(*.png)")
            # 如果文件存在
            if file_path:
                self.ui.FilenameBrowser.setText(file_path)
                self.img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                print(type(self.img))
                #  如果存在多通道的值（如RGB），需要转换为灰度图
                if len(self.img.shape) > 2:
                    self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

                # 对图片重新规划大小
                img_rows, img_cols = np.shape(self.img)
                aspect_ratio = np.double(img_rows) / np.double(img_cols)
                new_width = 200
                new_height = new_width / aspect_ratio
                self.img = cv2.resize(self.img, (int(new_width), int(new_height)))
                # cv2.imshow("", self.im)
                self.ImageDealClass.im = self.img
                # 显示输入的图片
                frame = QImage(self.img, int(new_width), int(new_height), QImage.Format_Grayscale8)
                pix = QPixmap.fromImage(frame)
                # pix=QPixmap(file_path)
                self.ui.ImageView.setPixmap(pix)
                self.ui.ImageView.setScaledContents(True)  # 让图片自适应label大小
                self.ui.BeginTestButton.setEnabled(1)
                self.ui.StepByStepBox.setEnabled(1)


        except Exception as e:
            QMessageBox.information(self, "错误", str(e))

    def EnhanceImage(self):
        """
        图像增强\n
        :return: None
        """
        try:
            self.ImageDealClass.image_enhance()

            cv2.imwrite('temp/Enhance.png', self.ImageDealClass.enhanceim)
            self.ui.EnhanceView.setPixmap(QPixmap('temp/Enhance.png'))
            self.ui.EnhanceView.setScaledContents(True)  # 让图片自适应label大小

            if self.ui.checkBox.isChecked():
                im = cv2.resize(self.ImageDealClass.norm_img, (400, 400))
                cv2.imshow("After Segement normalize", im)
                # im=cv2.resize(self.ImageDealClass.mask,(400,400))
                # cv2.imshow("Mask",im)
                im = cv2.resize(self.ImageDealClass.orientim, (400, 400))
                cv2.imshow("Directional map of the continuous distribution of fingerprints", im)


        except Exception as e:
            QMessageBox.information(self, "错误", str(e))

    def ThinImage(self):
        """
        图像细化\n
        :return: None
        """
        try:
            self.ImageDealClass.Thin(self.ImageDealClass.enhanceim)
            cv2.imwrite('temp/thin.png', self.ImageDealClass.ThinImg)
            self.ui.ThinView.setPixmap(QPixmap('temp/thin.png'))
            self.ui.ThinView.setScaledContents(True)  # 让图片自适应label大小
        except Exception as e:
            QMessageBox.information(self, "错误", str(e))

    def GetFeature(self):
        """
        获取特征\n
        :return: None
        """
        try:
            self.ImageDealClass.getFeature(self.ImageDealClass.ThinImg)
            cv2.imwrite('temp/feature.png', self.ImageDealClass.featureImg)
            self.ui.FeatureView.setPixmap(QPixmap('temp/feature.png'))
            self.ui.FeatureView.setScaledContents(True)  # 让图片自适应label大小

            self.ui.ResTextBrowser.clear()
            self.ui.ResTextBrowser.append("点坐标(x,y) 特征类型\t\t角度（°）\n")
            for i in range(len(self.ImageDealClass.features)):
                self.ui.ResTextBrowser.append(str(self.ImageDealClass.features[i]) + '\n')
            self.ui.SaveButton.setEnabled(1)
        except Exception as e:
            QMessageBox.information(self, "错误", str(e))

    def OneKeyTest(self):
        """
        一键测试\n
        :return: None
        """
        try:
            self.EnhanceImage()
            self.ThinImage()
            self.GetFeature()
        except Exception as e:
            QMessageBox.information(self, "错误", str(e))

    def SaveFeature(self):
        """
        保存特征文件\n
        :return: None
        """
        try:
            filepath, _ = QFileDialog.getSaveFileName(self, "选择保存特征文件路径", os.getcwd(), "txt(*.txt)")
            if filepath:
                with open(str(filepath), 'w') as f:
                    f.write("点坐标(x,y)\t特征类型\t\t角度（°）\n")
                    for feature in self.ImageDealClass.features:
                        for i in feature:
                            if i == "端点":
                                f.write(str(i) + "\t\t")
                            else:
                                f.write(str(i) + "\t")
                        f.write("\n")
                f.close()
                QMessageBox.information(self, "保存文件成功", "已成功保存特征文件在" + str(filepath))
            else:
                QMessageBox.information(self, "错误", "未选择特征文件保存路径！")
        except Exception as e:
            QMessageBox.information(self, "错误", str(e))


# 程序入口
if __name__ == '__main__':
    win = FingerPrintRecongnitionSystem()
    sys.exit(win.app.exec())
