from typing import Tuple
from PIL import Image
from torch.functional import Tensor
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid, save_image
import torch

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from model import CNN
import numpy as np

import matplotlib.pyplot as plt

def create_mask(size:Tuple):
    mask = Image.new('RGB', size, (0,0,0))
    return mask

def slide_mask(src_img : Tensor, mask_size : Tuple):
    mask = create_mask(mask_size)
    src_img_PIL = transforms.ToPILImage()(src_img)
    stride_r, stride_c = src_img.size()[0]//mask_size[0], src_img.size()[1]//mask_size[1]

    dst_imgs = []

    dst_imgs.append(src_img.unsqueeze(0).unsqueeze(0))

    for r in range(stride_r):
        for c in range(stride_c):
            dst = src_img_PIL.copy()
            dst.paste(mask, (c*mask_size[0], r*mask_size[1]))
            
            dst_tensor = to_tensor(dst).unsqueeze(0)
            dst_imgs.append(dst_tensor)

    dst_imgs = torch.cat(dst_imgs,dim=0)

    return dst_imgs, (stride_r, stride_c)

def calc_score(src : Tensor):
    score = []
    base_score = src[0]

    for i in range(1,len(src)):
        score.append(base_score - src[i])
    
    return np.array(score)

def get_colormap(src:np.ndarray, resize, cm_name = 'bwr'):
    cm = plt.get_cmap(cm_name)

    shift_index = np.hstack([
        np.linspace(src.min(), 0, 128, endpoint=False), 
        np.linspace(0, src.max(), 129, endpoint=True)
    ])

    src_shift = np.zeros_like(src)
    for r in range(src.shape[0]):
        for c in range(src.shape[1]):
            dist = (src[r,c] - shift_index)**2
            src_shift[r,c] = np.argmin(dist)

    cmap = cm(src_shift/255)
    cmap = (cmap * 255).astype(np.uint8)

    cmap = Image.fromarray(cmap)
    cmap = cmap.resize(resize)

    cmap.save('./res/test.png')



if __name__ == '__main__':
    # img = Image.open('./src/image.jpg')
    # img_resize = img.resize((256,256))
    # masked_img = slide_mask(img_resize, (32,32))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepareing MNIST dataset
    test_dataset = MNIST(root='./data', train=False, transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]))

    #Limit label
    test_mask = (test_dataset.targets == 0) | (test_dataset.targets == 6)
    test_dataset.data = test_dataset.data[test_mask]
    test_dataset.targets = test_dataset.targets[test_mask]
    test_dataset.targets[test_dataset.targets == 6] = 1

    test_loader = DataLoader(dataset=test_dataset, batch_size = 1, shuffle=False)

    model = CNN().to(device)
    model.load_state_dict(torch.load('./model/weights.pt'))

    with torch.no_grad():
        for x, labels in test_loader:
            x, (r,c)= slide_mask(x.squeeze(), (4,4))
            # x = torch.cat([x, x_mask],dim=0)

            x = x.to(device)

            outputs = model(x)
            outputs = outputs.to('cpu')
            outputs = outputs[:,0]

            scores = calc_score(outputs)
            scores = scores.reshape(r,c)

            get_colormap(scores, (x.size()[2],x.size()[3]))

