# Author: Yuru Jia. Last Modified: 18/12/2023

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

def get_connected_components(seg, comp_area_thre=30000, mode="large"):
    """
    从一个给定的图像分割掩码中(不是one-hot, 1-19这种mask)
    对于每个类别, 分别提取连通组件, 并根据组件的面积过滤不符合条件的连通组件
    例如comp_area_thre=30000, mode="large"
    意思则是保留大于(large)30000面积的物体, 小于这个面积的过滤掉

    最后返回的
    comps的格式, 是和分割mask一模一样的, 也是分割mask, 只是经过过滤罢了
    n_comps就代表这张图里连通分量的个数(不区分类别)
    
    args:
        seg: segmentation mask, [H, W], max(seg) = 255
    """
    n_comps = 1
    comps = np.zeros_like(seg)
    for c in np.unique(seg): 
        if c == 255:
            continue
        class_mask = (seg == c)
        class_mask = class_mask.astype(np.uint8)
        n_cls_comp, cls_comps, stats, centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        for c_i in range(1, n_cls_comp):
            area = stats[c_i][cv2.CC_STAT_AREA]
            if mode == "large":
                if area > comp_area_thre:
                    comps[cls_comps == c_i] = n_comps
                    n_comps += 1
            elif mode == "small":
                if area < comp_area_thre:
                    comps[cls_comps == c_i] = n_comps
                    n_comps += 1
    return comps, n_comps # 


def get_views(panorama_height, panorama_width, window_size=64, stride=16):
    """
    这个函数用于从全景图像中生成多个小视图（子窗口），每个视图都是一个更小的图像区块
    返回每一个view的坐标(w_start, h_start, w_end, h_end)
    """
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((w_start, h_start, w_end, h_end))
    return views


def encode_latents(image, vae, generator):
    """
    这个函数使用变分自编码器（VAE）对输入图像进行编码，
    获取其潜在表示（latent representation）
    """
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    image = to_tensor(image).unsqueeze(0).to("cuda", dtype=torch.float32)
    latents = vae.encode(image).latent_dist.sample(generator)
    latents = latents * vae.config.scaling_factor

    return latents