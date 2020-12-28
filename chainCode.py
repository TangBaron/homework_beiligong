import numpy as np
import pandas as pd
import cv2
from math import sqrt
from matplotlib import pyplot as plt
from itertools import chain
import argparse

#读取图像
def getchainCode(img, thresh):
    # 二值化图像
    ret, img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    plt.imshow(img, cmap='Greys')
    plt.show()

    # 查找要开始的第一个点
    for i, row in enumerate(img):
        for j, value in enumerate(row):
            if value == 255:
                start_point = (i, j)
                print(start_point, value)
                break
        else:
            continue
        break

    directions = [0, 1, 2,
                  7, 3,
                  6, 5, 4]
    dir2idx = dict(zip(directions, range(len(directions))))

    change_j = [-1, 0, 1,  # x or columns
                -1, 1,
                -1, 0, 1]

    change_i = [-1, -1, -1,  # y or rows
                0, 0,
                1, 1, 1]

    # 开始进行Freeman 列表算法
    border = []
    chain = []
    curr_point = start_point
    for direction in directions:
        idx = dir2idx[direction]
        new_point = (start_point[0] + change_i[idx], start_point[1] + change_j[idx])
        if img[new_point] != 0:  # if is ROI
            border.append(new_point)
            chain.append(direction)
            curr_point = new_point
            break

    count = 0
    while curr_point != start_point:
        # figure direction to start search
        b_direction = (direction + 5) % 8
        dirs_1 = range(b_direction, 8)
        dirs_2 = range(0, b_direction)
        dirs = []
        dirs.extend(dirs_1)
        dirs.extend(dirs_2)
        for direction in dirs:
            idx = dir2idx[direction]
            new_point = (curr_point[0] + change_i[idx], curr_point[1] + change_j[idx])
            if image[new_point] != 0:  # if is ROI
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
                break
        if count == 1000:
            break
        count += 1

    print(count)
    print(chain)

    plt.imshow(img, cmap='Greys')
    plt.plot([i[1] for i in border], [i[0] for i in border])

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=r"img\three.jpg")
    parser.add_argument("--threshold", type=int, default=70)
    hp = parser.parse_args()

    image = cv2.imread(hp.img, 0)
    getchainCode(image, hp.threshold)
