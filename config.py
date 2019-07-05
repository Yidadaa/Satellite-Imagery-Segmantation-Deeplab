'''
各种参数配置
'''

# 训练时图像大小
im_w = 480
im_h = 480

# dataloader参数
dataloader_config = {
    'batch_size': 128,
    'shuffle': True,
    'num_workers': 4,
    'pin_memory': True
}

# 数据集参数配置
dataset_config = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

# 训练时参数
train_config = {
    'max_epoch': 10,
    'lr': 1e-5,
    'log_interval': 2
}