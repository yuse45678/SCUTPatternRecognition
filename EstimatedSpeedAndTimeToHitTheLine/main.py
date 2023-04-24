# -*- coding = utf-8 -*-
# @time:2022/5/01 20:41
# Author:LeeJackson
# @File:main.py
# @Software:PyCharm

# 环境配置请查看requirement.txt文件！
# numpy==1.21.3
# opencv_python==4.5.5.62
# scikit_image==0.19.2

# 导入python模块
import math
import cv2
import cv2.cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def BackgroundSubtractorMOG(frame,kernel,model):
    """
    基于高斯混合模型的背景差分法，用于粗略识别运动中的小汽车\n
    :param frame: 待处理的图像帧
    :param kernel: cv2.getStructuringElement构造形态学使用的kernel
    :param model: 构造高斯混合模型
    :return:
    """
    (x, y, w, h)=(0, 0, 0, 0)
    # 运用高斯模型进行拟合，在两个标准差内设置为0，在两个标准差外设置为255
    fgmk = model.apply(frame)
    # 使用形态学的开运算做背景的去除
    fgmk = cv2.morphologyEx(fgmk, cv2.MORPH_OPEN, kernel)
    # cv2.findContours计算fgmk的轮廓
    contours, hierarchy = cv2.findContours(fgmk.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
    for c in contours:
        # 只有足够大的矩形框才能当作是小汽车
        if cv2.contourArea(c) < 8000:
            continue
        (x, y, w, h) = cv2.boundingRect(c) # 该函数计算矩形的边界框

    return (x, y, w, h)

def getP(frame,src_point,width,length):
    """
    获取线圈区域中的像素点，并且返回\n
    :param frame:当前帧
    :param src_point:虚拟线圈的起始点
    :param width:虚拟线圈的宽
    :param length:虚拟线圈的长
    :return: rec:虚拟线圈内部像素点
    """
    # 将当前帧转换为灰度图像
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 虚拟线圈的大小，（列数，行数）
    dsize=(length,width)
    # 虚拟线圈的起始位置
    src_point = np.float32(src_point)
    # 获取虚拟线圈的四个端点
    dst_point = np.float32([[0,0],[0,dsize[1]-1],[dsize[0]-1,dsize[1]-1],[dsize[0]-1,0]])
    # 至少要4个点，一一对应，找到映射矩阵h
    h, s = cv2.findHomography(src_point, dst_point, cv2.RANSAC, 10)
    # 获取线圈内部像素信息
    rec = cv2.warpPerspective(gray_img, h, dsize)

    return rec

def settingVirtualCoil(frame,startpX,startpY,rotateAnglePi,length,width):
    """
    设置虚拟线圈并且在当前帧显示\n
    :param frame:当前帧
    :param startpX:虚拟线圈起始点x坐标
    :param startpY:虚拟线圈起始点y坐标
    :param rotateAnglePi:线圈旋转角度，用弧度值表示
    :param length:虚拟线圈的长
    :param width:虚拟线圈的宽
    :return:虚拟线圈的四个端点
    """
    # 获取旋转角度的sin与cos值
    cosA=math.cos(rotateAnglePi)
    sinA=math.sin(rotateAnglePi)
    # 计算四个端点
    p1x= int(startpX)
    p1y= int(startpY)

    p2x= int(p1x+length*sinA)
    p2y= int(p1y-length*cosA)

    p3x= int(p2x)
    p3y= int(p2y - width)

    p4x= int(p1x)
    p4y= int(p1y-width)

    # 调用opencv库绘制线圈
    cv2.line(frame, (p1x, p1y), (p2x, p2y), (0, 255, 0), 2)
    cv2.line(frame, (p2x, p2y), (p3x, p3y), (0, 255, 0), 2)
    cv2.line(frame, (p3x, p3y), (p4x, p4y), (0, 255, 0), 2)
    cv2.line(frame, (p4x, p4y), (p1x, p1y), (0, 255, 0), 2)
    return [[p1x,p1y],[p2x,p2y],[p3x,p3y],[p4x,p4y]]

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # 调用opencv，读取视频
    cap = cv2.VideoCapture("TestVideo.mp4")
    # 初始化速度变量
    velocity = None
    # 初始化帧数变量
    framecount = 0
    # 计算终点线的斜率，并转换为角度
    anglePi=math.atan((488.0-533.0)/(600.0-96.0))
    # 初始化线圈长度与宽度
    VirtualCoilLength = 100
    VirtualCoilWidth = 20
    # 获取视频帧率
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    # cv2.getStructuringElement构造形态学使用的kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # 构造高斯混合模型，以待后续背景差分法所调用
    model = cv2.createBackgroundSubtractorMOG2()

    # ----------------------------视频处理程序---------------------------
    while cap.isOpened():# 如果视频被打开了就执行处理程序
        # 帧数+1
        framecount += 1
        # 读取一帧
        ret, frame = cap.read()
        if ret:
            # 将当前帧重置为宽高为600*800的画面
            frame = cv2.resize(frame, (600, 800))
            # cv2.imwrite('FirstFrame.jpg', frame)
            # break
            # cv2.line(frame,(439,336),(int(439+sinA*100),int(336-cosA*100)),(0,255,0),3)
            # cv2.line(frame,(329,283),(int(329+sinA*100),int(283-cosA*100)),(0,255,0),3)

            # ------------------目标检测模块------------------------------------
            (x, y, w, h) = BackgroundSubtractorMOG(frame,kernel,model)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # ------------------车速检测模块------------------------------------
            # 设置第一个线圈，起始点为（166，419）
            # 图像的y轴正方向向下，因此获取的角度需要加上pi/2
            src_point1= settingVirtualCoil(frame, 166, 419, math.pi / 2 + anglePi,
                                           VirtualCoilLength, VirtualCoilWidth)
            # 设置第二个线圈，起始点为（254，447）
            src_point2 = settingVirtualCoil(frame, 254, 447, math.pi / 2 + anglePi,
                                            VirtualCoilLength, VirtualCoilWidth)

            if framecount == 1: # 如果是第一帧就进行初始化
                passingFirstCoil = False  # 判断是否经过第一个线圈的标志
                passingSecondCoil = False  # 判断是否经过第而个线圈的标志
                # 获取没有车辆通过时第一个线圈像素点
                initRectangleOfA = getP(frame, src_point1, VirtualCoilWidth, VirtualCoilLength)
                # 获取没有车辆通过时第二个线圈像素点
                initRectangleOfB = getP(frame, src_point2, VirtualCoilWidth, VirtualCoilLength)
            else:
                # 判断虚拟线圈是否在目标车辆的框选矩形内部
                if x <= 166 and y + h >= 419 and x + w >= 166 + VirtualCoilLength and y <= 419 - VirtualCoilWidth:
                    # 获取当前帧第一个线圈中的像素点
                    newRectangleOfA = getP(frame, src_point1, VirtualCoilWidth, VirtualCoilLength)
                    # 调用SSIM方法来比较初始帧与当前帧的第一个线圈的差异
                    similarityOfA = ssim(initRectangleOfA, newRectangleOfA)
                    # 如果没有车辆通过第二个线圈就获取当前帧第二个线圈中的像素点
                    if not passingSecondCoil:
                        laterRectangleOfB = getP(frame, src_point2, VirtualCoilWidth, VirtualCoilLength)
                        similarityOfB = ssim(initRectangleOfB, laterRectangleOfB)
                    #print(similarityOfA)
                    # 如果第一个线圈的中相似度小于0.1，且两个标志均为false，证明车辆通过第一个线圈
                    if similarityOfA < 0.1 and not passingFirstCoil and not passingSecondCoil:
                        passingFirstCoil=True
                        start_frame = framecount
                        print("已经通过第一个线圈",start_frame)
                    # 如果第二个线圈中相似度小于0.1且通过第一个线圈的标志为True，则证明车辆正在通过第二个线圈
                    if similarityOfB < 0.1 and passingFirstCoil:
                        passingSecondCoil = True
                        end_frame = framecount
                        print("已通过第二个线圈",end_frame)
                        # 查阅国标可得，在小于30km/h的道路上箭头的长度为300cm即3m
                        velocity = 3/((end_frame-start_frame)/fps_video)
                        print("车速为",velocity,"m/s")
                        # 查阅国标可得，箭头到标线的距离在3m，标线宽为20cm
                        hit_time = 3.2/velocity
                        print("预计",hit_time,"秒后撞线")
                        passingFirstCoil = False


                # 这里是检查车辆是否完全通过第二个线圈，以避免重复检测
                if passingSecondCoil:
                    laterRectangleOfB2 = getP(frame, src_point2, VirtualCoilWidth, VirtualCoilLength)
                    SimilarityOfB=ssim(laterRectangleOfB, laterRectangleOfB2)
                    if SimilarityOfB < 0.1:
                        passingSecondCoil = False


                # ------------------ 在视频中显示相关信息----------------------------------------
                frame = cv2.putText(frame,"Fps: "+str('%.2f' %fps_video),(10,30),
                                  cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
                frame = cv2.putText(frame,"FrameCount: "+str(framecount),(10,55),
                                  cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
                frame = cv2.putText(frame,"push \"space\" to stop or start video",(10,80),
                                  cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
                frame = cv2.putText(frame,"push \"q\" to quit",(10,105),
                                  cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
                if passingFirstCoil:
                    frame = cv2.putText(frame, "The car has passed the first virtual coil", (10, 705),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                elif velocity is not None and passingSecondCoil:
                    frame = cv2.putText(frame, "The car has passed the second virtual coil", (10, 665),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    frame = cv2.putText(frame, "The car's speed is " + str('%.2f' % velocity) + "m/s ("+ str('%.2f' % (3.6*velocity))+"km/h)",
                                        (10, 685),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    frame = cv2.putText(frame, "The car will hit the line in " + str('%.2f' % hit_time) + " s", (10, 705),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("VehicleSpeedTestAndGuessHitTime", frame)

            # --------键盘控制视频---------------
            # 读取键盘值
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                # 设置Q按下时退出
                break
            elif key == ord(' '):
                # 设置空格按下时暂停或播放视频
                cv2.waitKey(0)

        else:
            break

    print(fps_video)
    cap.release()
    cv2.destroyAllWindows()


