# Author: Yuru Jia
# Last Modified: 2023-12-15

import os
import os.path as osp
import random
import logging
import argparse
import json

import numpy as np
import torch

from PIL import Image

from diffusers import DDIMScheduler,UniPCMultistepScheduler
from diffusers import AutoencoderKL

from controlnet.controlnet_model import ControlNetModel
from controlnet.tools.training_classes import (
    get_class_stacks, # 把输入图片中的类别信息提取成字符串
    make_one_hot, # 转成one-hot编码
    get_cs_classes, # 返回Cityscapes数据集中定义的类别名称列表
    map_label2RGB # 将输入的标签图（类别索引的二维数组）转换为彩色图像
    )
from controlnet.pipeline_refine import StableDiffusionControlNetRefinePipeline
from controlnet.tools.refine import get_connected_components, encode_latents


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet inference script.")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./example_data/output",
        help="output image save path",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--num_generated_images",
        type=int,
        default=1,
        help="Number of generated images per prompt.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Crop resolution.",
    )    
    parser.add_argument(
        "--gen_file",
        type=str,
        default="./example_data/gen_file.json",
        help="Label files to be generated.",
    )
    parser.add_argument(
        "--multidiffusion_rescale_factor",
        type=int,
        default=2,
        help="Rescale factor for multidiffusion."
    )
    parser.add_argument(
        "--comp_area_thre_la",
        type=int,
        default=30000,
        help="Minimum area for large connected components."
    )
    parser.add_argument(
        "--multi_scale",
        type=int,
        default=1,
        help="1: use connected components analysis, 0: no connected components analysis."
    )    
    parser.add_argument(
        "--multi_diff_stride",
        type=int,
        default=16,
        help="Stride for multi-diffusion."
    )
    parser.add_argument(
        "--weather_prompt",
        type=list, 
        default=["snowy", "rainy", "sunny", "foggy", "night"],
        help="Diversify prompts from perspective of weather conditions."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def visualize_generated_image_grid(img_gen, img_mask, resolution=512):
    """返回一个包含三个部分的可视化网格，
    以展示生成的图像、对应的标签掩码的彩色表示，以及这两者的混合图像
    """

    vis_grid = Image.new('RGB', (resolution * 3, resolution), (255, 255, 255))
    vis_grid.paste(img_gen, (0, 0))
    vis_grid.paste(Image.fromarray(map_label2RGB(img_mask)), (resolution, 0))
    vis_grid.paste(Image.blend(img_gen, Image.fromarray(map_label2RGB(img_mask)), 0.5), 
                   (resolution* 2, 0, resolution * 3, resolution))
 
    return vis_grid


def get_random_crop_rcs(label_map, c, rng, crop_size=512, resize_ratio=1.0):
    """
    稀有类裁切, 这跟omnitasker的randomcrop类似, 
    就是反复随机crop, 如果crop的区域内, GT中不存在指定的稀有类, 或者所占的比例不够高, 
    那就重新crop, 直至找到足够大小占比的稀有类
    """
    if isinstance(label_map, str):
        label_map = Image.open(label_map)

    label_map_arr = np.array(label_map) 
    indices = np.where(label_map_arr == c)
    w, h = label_map.size

    # resize image
    if resize_ratio != 1.0:
        label_map = label_map.resize((int(w * resize_ratio), int(h * resize_ratio)), Image.NEAREST)
        w, h = label_map.size
        indices = np.where(label_map_arr == c)


    for _ in range(10):
        # idx = np.random.randint(0, len(indices[0]) - 1)
        idx = rng.integers(0, len(indices[0]) - 1) # rng是随机数, 使得随机选取裁剪区域
        y, x = indices[0][idx], indices[1][idx]
        x1 = min(max(0, x - crop_size // 2), w -  crop_size)
        y1 = min(max(0, y - crop_size // 2), h - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        if np.sum(label_map_arr[y1:y2, x1:x2] == c) > 0.01 * crop_size * crop_size:
            break

    new_condition_img = label_map.crop((x1, y1, x2, y2))
    new_texts = get_class_stacks(new_condition_img) # 计算裁剪后的新的类别字符串信息

    crop_results = {
        "crop_coords": (x1, y1, x2, y2), # 裁剪区域
        "crop_condition_img": new_condition_img, # 裁剪后的图片
        "crop_texts": new_texts, # 返回裁剪后的新的类别字符串信息
    }    
        
    return crop_results


def main(args):
    args.num_inference_steps=15 # 减少迭代，加快推理

    huggingface_path= "../DGInStyle-huggingface/"
    stablediffusion_path = "../stable-diffusion-v1-5/"
    
    # 一次性生成全部
    # args.gen_file="/mnt/DGdataset/gta6000_rcs1e-2/rcs_sample_gta_files/real_rcs_images_6000_segment.json"
    # args.output_folder="/mnt/DGdataset/dginstyle_gen_segment"
    
    # 分成几个生成
    segment_id=2
    args.gen_file="/mnt/DGdataset/gta6000_rcs1e-2/rcs_sample_gta_files/real_rcs_images_6000_segment_{}.json".format(segment_id)
    args.output_folder="/mnt/DGdataset/dginstyle_gen_segment_{}".format(segment_id)
    print(args.output_folder)

    controlnet = ControlNetModel.from_pretrained(huggingface_path+"ControlNet_UNet-S",
                    #torch_dtype=torch.float16, # 改成fp16
                    subfolder="ControlNet_UNet-S", revision=None)    
    # prepare the model and the pipeline    
    vae = AutoencoderKL.from_pretrained(
                     stablediffusion_path, 
                     #torch_dtype=torch.float16, # 改成fp16
                     subfolder="vae", revision=None)
    vae.requires_grad_(False)
    vae.to("cuda", dtype=torch.float32)
    #vae.to("cuda", dtype=torch.float16) # 改成fp16

    pipe = StableDiffusionControlNetRefinePipeline.from_pretrained(
                    stablediffusion_path, 
                    controlnet=controlnet)
    
    #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    pipe.enable_xformers_memory_efficient_attention() # 启用 FlashAttention

    gen_seed = 0 # 随机种子是固定的，所以主要weather一致，生成图片就是一致的
    generator = torch.manual_seed(gen_seed)
    rng = np.random.default_rng(gen_seed)

    # create output folder
    out_dir = args.output_folder
    os.makedirs(out_dir, exist_ok=True)
    out_dir_img= osp.join(out_dir, "images")
    os.makedirs(out_dir_img, exist_ok=True)
    out_dir_label= osp.join(out_dir, "labels")
    os.makedirs(out_dir_label, exist_ok=True)
    out_dir_vis= osp.join(out_dir, "vis")
    os.makedirs(out_dir_vis, exist_ok=True)

    # config logging file
    logging.basicConfig(filename=os.path.join(out_dir, f"gen.log"), level=logging.INFO)

    # Read label file as input
    with open(args.gen_file, 'r') as of:
        gen_label_files = json.load(of)   # 取出gen_file.json中要生成图像的GT数据
        

    # 遍历每个要生成图像的GT数据
    print("Start generating...")
    total_files= len(gen_label_files)
    for i in range(total_files):
        print(f"Processing file {i + 1} of {total_files}")
        
        c = int(gen_label_files[i]["class"]) # 取出稀有类
        label_map = gen_label_files[i]["file_name"]  # 取文件名（路径）
        label_file_id = os.path.basename(label_map).replace("_labelTrainIds.png", "") # 取出图片ID
        label_map = Image.open(label_map) # 读取GT图片
        
        # get the original condition image which is cropped from the original label map with crop size 512
        # 获取裁剪后的图片，大小为512
        rcs_crop_results = get_random_crop_rcs(label_map, c, rng, crop_size=args.resolution)
        """
        包含：
        "crop_coords": (x1, y1, x2, y2), # 裁剪区域
        "crop_condition_img": new_condition_img, # 裁剪后的图片
        "crop_texts": new_texts, # 返回裁剪后的新的类别字符串信息
        """
        crop_condition_img = np.array(rcs_crop_results["crop_condition_img"]) # 裁剪后GT
        prompt = rcs_crop_results["crop_texts"] # 裁剪后GT的新prompt
        

        """
        这里是以50%的概率随机从args中指定的天气列表获取一个天气作为天气prompt
        """
        weather_prompt = None
        if np.random.rand() < 0.5 and args.weather_prompt is not None:
            weather_prompt = np.random.choice(args.weather_prompt)
            prompt = "A city street scene with " + prompt + ", in " + weather_prompt + " weather."
        logging.info(f"{label_file_id}, rcs_class: {get_cs_classes()[c]}, weather: {weather_prompt}, gen_seed: {gen_seed}")
        
        rescale_factor = args.multidiffusion_rescale_factor # paper中提到的东西

        # 连通组件分析
        # connected_components analysis               
        if args.multi_scale > 0: # args.multi_scale 大于0，则进入下面的代码块
            # Large connected components
            condition_comps_la, n_condition_comps = get_connected_components(
                crop_condition_img, 
                comp_area_thre=args.comp_area_thre_la, # 大型连通组件的最小面积:30000
                mode="large"
            )
            """
            condition_comps_la的就是过滤掉小物体的分割mask
            n_condition_comps就代表这张图里连通分量的个数(不区分类别)
            """
            components_mask_la = condition_comps_la!=0 
            components_mask_la = components_mask_la.astype(int)
            components_mask_la = torch.Tensor(components_mask_la)    
            """
            通过这个components_mask_la!=0 把1-19的像素都转成了1
            最后components_mask_l转成一个代表大型物体的二值掩码
            """

        # 把裁剪后的GT转成one-hot编码
        # process cropped image label into one-hot encoding
        crop_condition_img_onehot = torch.Tensor(make_one_hot(crop_condition_img))
        crop_condition_img_onehot = torch.unsqueeze(crop_condition_img_onehot.permute(2, 0, 1), 0) #[1, class, H, W]
                
        # initial generation
        images_ini = pipe(
            prompt, 
            num_inference_steps=args.num_inference_steps, # infer迭代次数
            generator=generator,  
            num_images_per_prompt=args.num_generated_images,# 每个prompt生成一张图
            cond_image=crop_condition_img_onehot, # 裁剪后的GT作为controlnet的condition图
            output_type="both",
            strength=1,
            rescale_factor=1,
            multi_diff_stride=64,
        ).images
        
        """
        得到初始生成图片images_ini
        """

        image_ini = images_ini["image"][0]
        # resize初始生成图images_ini
        image_ini_upsampl = image_ini.resize(
            (args.resolution*rescale_factor, args.resolution*rescale_factor), 
            Image.LANCZOS)
        # 把初始生成图images_ini编码到潜在空间
        image_ini_upsampl_latents = encode_latents(image_ini_upsampl, vae, generator) 

        # Multi-diffusion generation with large components impainting
        images = pipe(
            prompt, 
            num_inference_steps=args.num_inference_steps, 
            generator=generator, 
            num_images_per_prompt=args.num_generated_images,
            cond_image=crop_condition_img_onehot, # 裁剪后的GT作为controlnet的condition图
            output_type="pil", 
            strength=1,
            # rescale_factor=2  Rescale factor for multidiffusion 
            # 这个参数指定是否为mutilscale-diffusion
            rescale_factor=rescale_factor, 
            multi_diff_stride=args.multi_diff_stride, # args.multi_diff_stride=16
            add_inpaint=True, 
            ini_latents=image_ini_upsampl_latents,# 初始生成的图片images_ini(潜在空间编码)
            init_img_mask_la=components_mask_la # 大型物体的二值掩码       
        ).images
        """
        再次生成，得到最终生成图片
        """

        # 检查生成的第一张图像是否具有所需的分辨率
        # 如果没有, 使用Image.LANCZOS滤波器调整图像大小至指定分辨率
        # Image.LANCZOS滤波器以在缩小图像时产生高质量结果而闻名
        if images[0].size != (args.resolution, args.resolution):
            images[0] = images[0].resize(
                (args.resolution, args.resolution), 
                Image.LANCZOS)
        
        # 保存结果
        output_file_img = f"{out_dir_img}/{label_file_id}_genid{gen_seed}.png"
        output_file_label = f"{out_dir_label}/{label_file_id}_genid{gen_seed}_labelTrainIds.png"
        output_file_grid = f"{out_dir_vis}/{label_file_id}_genid{gen_seed}.png"
        
        images[0].save(output_file_img)

        if not isinstance(crop_condition_img, Image.Image):
            crop_condition_img = Image.fromarray(crop_condition_img)
        crop_condition_img.save(output_file_label)


        # 返回生成的图像、对应的标签掩码的彩色表示，以及这两者的混合图像
        vis_grid = visualize_generated_image_grid(images[0], crop_condition_img, resolution=args.resolution)
        vis_grid.save(output_file_grid)


if __name__ == "__main__":
    args = parse_args()  
    main(args)

