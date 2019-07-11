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
    'mean': [0.5231372549019608, 0.5218039215686274, 0.4533333333333333],
    'std': [0.23662745098039217, 0.2096470588235294, 0.20380392156862745]
}

# 训练时参数
train_config = {
    'max_epoch': 30,
    'lr': 1e-4,
    'log_interval': 2
}