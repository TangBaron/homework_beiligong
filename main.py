import cv2 as cv
import numpy as np

ans = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]#采用8连通区域标记

def dfs(img, result, i, j, now):
    if i < 0 or i == img.shape[0] or j < 0 or j >= img.shape[1]:
        return
    if img[i][j] == 0 or result[i][j] > 0:
        return
    if img[i][j] == 255 and result[i][j] == 0:
        result[i][j] = now
    for x in range(8):
        dfs(img, result, i + ans[x][0], j + ans[x][1], now)

def region_label(img):
    shape = img.shape
    result = np.empty(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            result[i][j] = 0
    now = 1#当前标记
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img[i][j] == 255 and result[i][j] == 0:#遇到一个还未标记过的白色区域
                dfs(img, result, i, j, now)
                now = now + 1
    return result

def getArea(img):
    total = img.shape[0] * img.shape[1]
    print("总像素：{}".format(total))
    result = dict()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] not in result:
                result[img[i][j]] = 0
                for p in range(img.shape[0]):
                    for q in range(img.shape[1]):
                        if img[p][q] == img[i][j]:
                            result[img[i][j]] = result[img[i][j]] + 1
    for item in result:
        print("区域编号 {} 像素面积 {} 像素".format(item, result[item]))

if __name__ == "__main__":
    img = np.loadtxt("img/source.txt")
    for i in range(10):
        for j in range(10):
            if img[i][j] == 1:
                img[i][j] = 255
    cv.imwrite("ressult/source.jpg", img)
    result = region_label(img)
    print(result)
    getArea(result)