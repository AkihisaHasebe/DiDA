from typing import Tuple
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid, save_image
import torch

def create_mask(size:Tuple):
    mask = Image.new('RGB', size, (0,0,0))
    return mask

def slide_mask(src_img : Image.Image, mask_size : Tuple):
    mask = create_mask(mask_size)

    stride_r, stride_c = src_img.size[0]//mask_size[0], src_img.size[1]//mask_size[1]

    dst_imgs = []

    for r in range(stride_r):
        for c in range(stride_c):
            dst = src_img.copy()
            dst.paste(mask, (c*mask_size[0], r*mask_size[1]))
            
            dst_tensor = to_tensor(dst).unsqueeze(0)
            dst_imgs.append(dst_tensor)

    dst_imgs = torch.cat(dst_imgs,dim=0)

    return dst_imgs

if __name__ == '__main__':
    img = Image.open('./src/image.jpg')
    img_resize = img.resize((256,256))
    masked_img = slide_mask(img_resize, (32,32))

    grid_imgs = make_grid(masked_img)
    print(grid_imgs.shape)

    save_image(grid_imgs, './res/result.png')