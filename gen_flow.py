import sys
sys.path.append('core')
import re
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from get_var import get_var

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder
import matplotlib.pyplot as plt
from interpolations import warp_flow


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(imfile):
    img = cv2.imread(imfile)#np.array(Image.open(imfile)).astype(np.uint8)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo,i):
    # img = img[0].permute(1,2,0).cpu().numpy()
    # flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)


    plt.imshow(img_flo / 255.0)
    plt.savefig(f"opt_flow{i+1}.jpg")
    plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model,map_location=torch.device('cpu')))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg')) + \
                     glob.glob(os.path.join(args.path, '*.pgm')) + \
                         glob.glob(os.path.join(args.path, '*.ppm'))
        # images = sorted(images)
        images.sort(key=lambda f: int(re.sub("\D", "", f)))
        
        if args.path == 'Sphere':
            N = len(images)-2
        else:
            N = len(images)-1
        
        for ind in range(0,N,2):
        #     image1 = cv2.imread(images[ind])
        #     image2 = cv2.imread(images[ind+2])
        # for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(images[ind])
            image2 = load_image(images[ind+2])

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            # flow_up = flow_up[0].permute(1,2,0).cpu().numpy()
            # f_flow, If = get_var(image1,image2, flow_up)
            # breakpoint()
            flow_low, flow_up = model(image2, image1, iters=20, test_mode=True) # small - 990162
            flow_up = flow_up[0].permute(1,2,0).cpu().numpy()
            flow, I = get_var(image2,image1,flow_up)
            image1 = image1[0].permute(1,2,0).cpu().numpy()
            image2 = image2[0].permute(1,2,0).cpu().numpy()
            
            warp_flow(image2,flow,ind)
            
            # viz(image1, flow_up, ind)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)