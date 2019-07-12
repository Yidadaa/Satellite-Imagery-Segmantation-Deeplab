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

# 数据集参数配置
dataset_config = {
    'mean': [133.4 , 133.06, 115.6],
    'std': [60.34, 53.46, 51.97]
}

# 训练时参数
train_config = {
    'max_epoch': 30,
    'lr': 1e-4,
    'log_interval': 2
}