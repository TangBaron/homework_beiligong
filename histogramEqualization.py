import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

#计算像素值出现的概率
def calProbability(img, Level):
    '''
    :param img: 图像输入的numpy 矩阵
    :return:
    '''

    prob = np.zeros(shape=(Level))

    for row in img:
        for col in row:
            prob[col] += 1

    return prob/(img.shape[0] * img.shape[1])

#根据概率进行直方图均衡化
def probability2Histogram(img, prob, Level):
    '''
    :param img:  图像输入的numpy矩阵
    :param prob: 图像输入的概率矩阵
    :return: 直方图均衡化后的图像
    '''
    prob = np.cumsum(prob) #累计概率
    img_map = Level * prob
    row, col = img.shape
    for r in range(row):
        for c in range(col):
            img[r, c] = img_map[img[r, c]]
    return img

#绘制直方图
def plot(y, Level):
    plt.figure()
    plt.bar([i for i in range(Level)], y, width=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=r"img\car.jpg")
    parser.add_argument("--Level", type=int, default=256)
    hp = parser.parse_args()

    img = cv2.imread(hp.img, 0)
    prob = calProbability(img, hp.Level)
    plot(prob, hp.Level)

    #直方图均衡化
    img = probability2Histogram(img, prob, hp.Level)
    cv2.imwrite(r"result\\His\\"+hp.img.split('\\')[-1], img)
    prob = calProbability(img, hp.Level)
    plot(prob, hp.Level)

    plt.show()






