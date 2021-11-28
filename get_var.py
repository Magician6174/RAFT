import numpy as np
import cv2


def get_var(img1, img2, opt_flow):
    img1 = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    fx = cv2.Sobel(img1,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
    fy = cv2.Sobel(img1,ddepth=cv2.CV_32F,dx=0,dy=1,ksize=-1)
    fx = cv2.convertScaleAbs(fx)
    fy = cv2.convertScaleAbs(fy)
    kernel_1 = np.array([[1,1],[1,1]])
    kernel_2 = np.array([[-1,-1],[-1,-1]])
    f1 = cv2.filter2D(img1,ddepth=-1,kernel=kernel_2)
    f2 = cv2.filter2D(img2,ddepth=-1,kernel=kernel_1)
    ft = f1 + f2
    flow = [opt_flow[:,:,0], opt_flow[:,:,1]]
    I = [fx, fy, ft]
    return flow, I