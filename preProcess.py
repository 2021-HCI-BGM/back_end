import cv2

import numpy as np


# from cv2.cv import CreateImage


def zh_ch(string):
    return string.encode("gbk").decode(errors="ignore")


def contrastStretch(srcImage):
    resultImage = srcImage  # 复制一个备份。
    nRows, nCols = resultImage.shape
    # 图像连续性判断
    # if resultImage.isContinuous():
    #     nCols = nCols * nRows
    #     nRows = 1
    # 图像指针操作
    pixMax = 0
    pixMin = 255  # 计算图像的最大最小值
    for j in range(nRows):
        for i in range(nCols):
            if resultImage[j, i] > pixMax:
                pixMax = resultImage[j, i]
            if resultImage[j, i] < pixMin:
                pixMin = resultImage[j, i]
    # 对比度拉伸映射
    for j in range(nRows):
        for i in range(nCols):
            resultImage[j, i] = (resultImage[j, i] - pixMin) * 255 / (pixMax - pixMin)
    return resultImage


def get_red(img):
    redImg = img[:, :, 2]
    return redImg


def get_green(img):
    greenImg = img[:, :, 1]
    return greenImg


def get_blue(img):
    blueImg = img[:, :, 0]
    return blueImg


def WhiteBalance(srcImage):
    resultImage = srcImage  # 复制一个备份。
    row, col, chn = resultImage.shape
    # row, col = srcImage.shape
    B = 0
    G = 0
    R = 0

    dst = np.zeros((row, col, 3), np.uint8)
    # 统计RGB
    for i in range(row):
        for j in range(col):
            B += 1.0 * srcImage[i, j, 0]
            G += 1.0 * srcImage[i, j, 1]
            R += 1.0 * srcImage[i, j, 2]
    # bgr的平均
    B /= (row * col)
    G /= (row * col)
    R /= (row * col)
    # printf("%.5f %.5f %.5f\n", B, G, R)
    # 计算灰度值
    GrayValue = (B + G + R) / 3
    # printf("%.5f\n", GrayValue)
    # 三个增益系数
    kr = GrayValue / R
    kg = GrayValue / G
    kb = GrayValue / B
    # printf("%.5f %.5f %.5f\n", kb, kg, kr)
    for i in range(row):
        for j in range(col):
            dst[i, j, 0] = 255 if (kb * int(srcImage[i, j, 0]) > 255) else int(kb * srcImage[i, j, 0])  # 做切割，放缩到255之内
            dst[i, j, 1] = 255 if ((int)(kg * srcImage[i, j, 1]) > 255) else (int)(kg * srcImage[i, j, 1])
            dst[i, j, 2] = 255 if ((int)(kr * srcImage[i, j, 2]) > 255) else (int)(kr * srcImage[i, j, 2])
            # dst[i, j,0] = (int)(kb * srcImage[i, j,0]) > 255 ? 255 : (int)(kb * [i, j,0])
            # dst[i, j,1] = (int)(kg * srcImage[i, j,1]) > 255 ? 255 : (int)(kg * [i, j,1])
            # dst[i, j,2] = (int)(kr * srcImage[i, j,2]) > 255 ? 255 : (int)(kr * [i, j,2])
    return dst


def skinDetectionHSV(pImage, lower, upper):
    row, col, chn = pImage.shape
    pImageHSV = np.zeros((row, col, 3), np.uint8)
    pImageProcessed = np.zeros((row, col, 1), np.uint8)
    pyrImage = np.zeros((int(row / 2), int(col / 2), 1), np.uint8)
    cv2.cvtColor(pImage, cv2.COLOR_BGR2HSV, pImageHSV)
    # cv2.imshow("HSV", pImageHSV)
    cv2.inRange(pImageHSV, (lower, smin, vmin), (upper, smax, vmax), pImageProcessed)
    # cv2.imshow("processed", pImageProcessed)
    cv2.pyrDown(pImageProcessed, pyrImage)
    cv2.pyrUp(pyrImage, pImageProcessed)
    # 矩形: MORPH_RECT
    # 交叉形: MORPH_CROSS
    # 椭圆形: MORPH_ELLIPSE
    erosion_type = cv2.MORPH_RECT
    erosion_size = 2
    element = cv2.getStructuringElement(erosion_type,
                                        (2 * erosion_size + 1, 2 * erosion_size + 1),
                                        (erosion_size, erosion_size))  # 中间点作为锚点
    cv2.erode(pImageProcessed, element, pImageProcessed, (-1, -1), 2)  # 腐蚀两次
    cv2.dilate(pImageProcessed, element, pImageProcessed, (-1, -1), 1)
    return pImageProcessed

def preprocess(srcImg):

    kernelSize = 3
    # cap = cv2.VideoCapture(0)  # 调整参数实现读取视频或调用摄像头

    # cv2.imshow("cap", srcImg)
    # 反转
    cv2.flip(srcImg, 1, srcImg)
    # cv2.imshow('after flip', srcImg)

    # 1.归一化
    # srcImage_B, srcImage_G, srcImage_R = cv2.split(srcImg)
    # # 0通道为B分量，1通道为G分量，2通道为R分量。因为：RGB色彩空间在opencv中默认通道顺序为BGR！！！
    # srcImage_B = contrastStretch(srcImage_B)
    # srcImage_G = contrastStretch(srcImage_G)
    # srcImage_R = contrastStretch(srcImage_R)
    # dest_BGR = cv2.merge([srcImage_B, srcImage_G, srcImage_R])
    # 考虑到效率问题只有用normalize...
    cv2.normalize(srcImg, srcImg, 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow('NORMALIZE', srcImg)

    # 2.平滑，去噪——滤波
    g_nKernelValue = kernelSize * 2 + 1
    # 高斯滤波
    cv2.GaussianBlur(srcImg, (g_nKernelValue, g_nKernelValue), 0, srcImg)
    # 均值滤波
    # cv2.blur(srcImg, (g_nKernelValue, g_nKernelValue), srcImg)
    # 中值滤波
    # cv2.medianBlur(srcImg, g_nKernelValue, srcImg)
    # 开操作去噪
    # 矩形: MORPH_RECT
    # 交叉形: MORPH_CROSS
    # 椭圆形: MORPH_ELLIPSE
    # erosion_type = cv2.MORPH_RECT
    # erosion_size = 2
    # element = cv2.getStructuringElement(erosion_type,
    #                                 (2 * erosion_size + 1, 2 * erosion_size + 1),
    #                                 (erosion_size, erosion_size))  # 中间点作为锚点
    # cv2.erode(srcImg, element, srcImg)  # 第三个参数是一个默认的kernel3 * 3
    # cv2.dilate(srcImg, element, srcImg)  # 第三个参数是一个默认的kernel3 * 3

    # cv2.imshow('blur', srcImg)
    # 白平衡
    # srcImg=WhiteBalance(srcImg)
    # cv2.imshow('WhiteBalance', srcImg)

    # 肤色检测
    mask = skinDetectionHSV(srcImg, 0, 40)
    # 颜色范围26~34 是黄色，也就是肤色 # 反正参数都可以调整...
    # 此时得到的只是掩膜，还需要做mask操作！！
    # 而且mask还需要拓展需要膨胀或者闭操作。
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    cv2.morphologyEx(mask, cv2.MORPH_DILATE, element, mask, (-1, -1), 1)  # 只是膨胀
    # cv2.imshow("after dilate", mask)
    dst = np.zeros(srcImg.shape, np.uint8)
    rows, cols, chn = srcImg.shape
    for i in range(rows):
        for j in range(cols):
            if mask[i, j] == 255:
                dst[i, j, 0] = srcImg[i, j, 0]
                dst[i, j, 1] = srcImg[i, j, 1]
                dst[i, j, 2] = srcImg[i, j, 2]

    # cv2.imshow("all", dst)
    # 最后决定退出
    # cap.release()
    # cv2.destroyAllWindows()
    return dst


# 色相
hmin = 0  # h分量取下限
hmax = 180  # h分量取上限
h_Max = 180  # h分量可取的最大值
# 饱和度
smin = 0  # s分量取下限
smax = 255  # s分量取上限
s_Max = 255  # s分量可取的最大值
# 亮度
vmin = 0  # v分量取下限
vmax = 255  # v分量取上限
v_Max = 255  # v分量可取的最大值

kernelSize = 3

# cap = cv2.VideoCapture(0)  # 调整参数实现读取视频或调用摄像头
# while 1:
#     ret, srcImg = cap.read()
#     # cv2.imshow("cap", srcImg)
#
#     # 反转
#     cv2.flip(srcImg, 1, srcImg)
#     cv2.imshow('after flip', srcImg)
#
#     # 1.归一化
#     # srcImage_B, srcImage_G, srcImage_R = cv2.split(srcImg)
#     # # 0通道为B分量，1通道为G分量，2通道为R分量。因为：RGB色彩空间在opencv中默认通道顺序为BGR！！！
#     # srcImage_B = contrastStretch(srcImage_B)
#     # srcImage_G = contrastStretch(srcImage_G)
#     # srcImage_R = contrastStretch(srcImage_R)
#     # dest_BGR = cv2.merge([srcImage_B, srcImage_G, srcImage_R])
#     # 考虑到效率问题只有用normalize...
#     cv2.normalize(srcImg, srcImg, 0, 255, cv2.NORM_MINMAX)
#     cv2.imshow('NORMALIZE', srcImg)
#
#     # 2.平滑，去噪——滤波
#     g_nKernelValue = kernelSize * 2 + 1
#     # 高斯滤波
#     cv2.GaussianBlur(srcImg, (g_nKernelValue, g_nKernelValue), 0, srcImg)
#     # 均值滤波
#     # cv2.blur(srcImg, (g_nKernelValue, g_nKernelValue), srcImg)
#     # 中值滤波
#     # cv2.medianBlur(srcImg, g_nKernelValue, srcImg)
#     # 开操作去噪
#     # 矩形: MORPH_RECT
#     # 交叉形: MORPH_CROSS
#     # 椭圆形: MORPH_ELLIPSE
#     # erosion_type = cv2.MORPH_RECT
#     # erosion_size = 2
#     # element = cv2.getStructuringElement(erosion_type,
#     #                                 (2 * erosion_size + 1, 2 * erosion_size + 1),
#     #                                 (erosion_size, erosion_size))  # 中间点作为锚点
#     # cv2.erode(srcImg, element, srcImg)  # 第三个参数是一个默认的kernel3 * 3
#     # cv2.dilate(srcImg, element, srcImg)  # 第三个参数是一个默认的kernel3 * 3
#
#     cv2.imshow('blur', srcImg)
#     # 白平衡
#     # srcImg=WhiteBalance(srcImg)
#     cv2.imshow('WhiteBalance', srcImg)
#
#     # 肤色检测
#     mask = skinDetectionHSV(srcImg, 0, 40)
#     # 颜色范围26~34 是黄色，也就是肤色 # 反正参数都可以调整...
#     # 此时得到的只是掩膜，还需要做mask操作！！
#     # 而且mask还需要拓展需要膨胀或者闭操作。
#     element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
#     cv2.morphologyEx(mask, cv2.MORPH_DILATE, element, mask, (-1, -1), 1)  # 只是膨胀
#     cv2.imshow("after dilate", mask)
#     dst = np.zeros(srcImg.shape, np.uint8)
#     rows, cols, chn = srcImg.shape
#     for i in range(rows):
#         for j in range(cols):
#             if mask[i, j] == 255:
#                 dst[i, j, 0] = srcImg[i, j, 0]
#                 dst[i, j, 1] = srcImg[i, j, 1]
#                 dst[i, j, 2] = srcImg[i, j, 2]
#
#     cv2.imshow("all", dst)
#     # 最后决定退出
#     if cv2.waitKey(100) & 0xff == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
