# ---------------------------------------------------------------
# Author Yuru Jia. Last modified: 05/01/2024
# ---------------------------------------------------------------

# dataset settings
dataset_type = 'CityscapesDataset'
data_root =  '/mnt/dataset/cityscapes/'#'data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
gen_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=(1280, 720)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75)
    dict(type='RandomFlip', prob=0.5),
    dict(type='BilateralFilter', prob=0.5),# 新增双边滤波器
    dict(type='PixMix', mixing_set_path='/mnt/dataset/fractals_and_fvis/fractals', img_scale=crop_size),# 新增pixMix
    #dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
gta_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 720)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='BilateralFilter', prob=0.5), # 新增双边滤波器
    dict(type='PixMix', mixing_set_path='/mnt/dataset/fractals_and_fvis/fractals', img_scale=crop_size),# 新增pixMix
#     dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
# 4090的话，显存和GPU利用率都才一半，改一改
#     samples_per_gpu=2,
#     workers_per_gpu=4,
    samples_per_gpu=4, # 决定是每个GPU上的batchsize
    workers_per_gpu=8, # 决定每个GPU分配的线程数
    train=dict(
        type='DGDataset',
        source2=dict(
            type='GTADataset',
            data_root='/mnt/dataset/dginstyle_gen/', #'data/dginstyle_gen/'
            img_dir='images',
            ann_dir='labels',
            pipeline=gen_train_pipeline),
        source=dict(
            type='GTADataset',
            data_root='/mnt/dataset/gta6000_rcs1e-2/', # 'data/gta6000_rcs1e-2/'
            img_dir='images',
            ann_dir='labels',
            pipeline=gta_train_pipeline)),
    val=dict(
        type='CityscapesDataset',
        data_root='/mnt/dataset/cityscapes/', # 'data/cityscapes/'
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type='CityscapesDataset',
        data_root='/mnt/dataset/cityscapes/', # 'data/cityscapes/'
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))

