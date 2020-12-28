import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt


# 拉普拉斯实现图像锐化
def laplacian_sharpening(img, K, K_size=3):
    H, W = img.shape
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
    tmp = out.copy()
    for y in range(H):
        for x in range(W):
            out[pad + y, pad + x] = (-1) * np.sum(K * (tmp[y: y + K_size, x: x + K_size])) + tmp[pad + y, pad + x]
    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=r"img\car.jpg")
    parser.add_argument("--mode", type=str, default="edge")
    hp = parser.parse_args()

    if hp.mode == 'sharp':
        img = cv2.imread(hp.img, 0).astype(np.float)
        plt.imshow(img, cmap='Greys')
        K =  [[0,1,0],[1,-4,1],[0,1,0]]
        out = laplacian_sharpening(img, K)
        plt.imshow(out, cmap='Greys')
        cv2.imwrite(r"result\\Lap\\sharp\\"+hp.img.split('\\')[-1], out)
        plt.show()
    elif hp.mode == 'edge':
        img = cv2.imread(hp.img, 0).astype(np.float)
        plt.imshow(img, cmap='Greys')
        K = [[1,1,1],[1,-8,1],[1,1,1]]
        out = laplacian_sharpening(img, K)
        plt.imshow(out, cmap='Greys')
        cv2.imwrite(r"result\\Lap\\edge\\" + hp.img.split('\\')[-1], out)
        plt.show()



