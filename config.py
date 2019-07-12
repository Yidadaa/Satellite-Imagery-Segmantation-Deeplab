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
    'mean': [118.20678319327652,143.28537851262593,139.68799169449989],
    'std': [49.766278381728014,49.24727857667724,57.03856461061226]
}

# dataset_config = {
#     'mean': [0, 0, 0],
#     'std': [1, 1, 1]
# }

# 训练时参数
train_config = {
    'max_epoch': 20,
    'lr': 1e-3
}