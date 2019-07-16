'''
各种参数配置
'''

# 训练时图像大小
im_w = 224
im_h = 224

# 类别
num_classes = 4

# dataloader参数
dataloader_config = {
    'batch_size': 96,
    'shuffle': True,
    'num_workers': 4,
    'pin_memory': True,
    'drop_last': True
}

dataset_config = {
    'mean': [0.45666892, 0.52336701, 0.52904614],
    'std': [0.19727931, 0.20078927, 0.23028788],
    'use_jitter': True
}