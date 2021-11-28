import cv2
import numpy as np
import scipy.interpolate
from matplotlib import pyplot as plt

def warp_flow(image, flow,ind):

    image = image[:, :, 0]
    height, width = image.shape

    u, v = flow

    xx, yy = np.meshgrid(np.arange(width),np.arange(height))

    img_for_x = xx + u
    img_for_y = yy + v

    xt = np.clip(img_for_x, 0, height - 1)
    yt = np.clip(img_for_y, 0, width - 1)

    It = np.zeros(image.shape)
    image1_interp = scipy.interpolate.RectBivariateSpline(np.arange(width), np.arange(height), image.T)

    for i in range(It.shape[0]):
        for j in range(It.shape[1]):
            It[i, j] = image1_interp(xt[i, j], yt[i, j])

    It = It.astype(np.int)
    # plt.subplot(1,2,1)
    # plt.imshow(image,cmap = 'gray')
    # plt.title("Given Image")
    # plt.subplot(1,2,2)
    plt.imshow(It,cmap = "gray")
    plt.title("interpolated frame")
    plt.savefig(f"interpolated_sphere{ind+1}.png")
    plt.show()
    return It

