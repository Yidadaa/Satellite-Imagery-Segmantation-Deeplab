'''
用于存放评估脚本
'''

import torch
from torch.functional import F
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

import os
import argparse

from model import DeepLabV3Res101
from utils import setup, restore_from, get_metrics
from dataloader import SegDataset
import config

def run_on_single_image(img_path:str, ckpt:str):
    '''对单张图片进行处理

    Args:
        img_path(str): 图片路径
        ckpt(str): 检查点路径
    '''
    assert img_path is not None and os.path.exists(img_path), '图片路径不存在: {}'.format(img_path)
    assert ckpt is not None and os.path.exists(ckpt), '检查点路径不存在: {}'.format(ckpt)

    print('Setting up model.')
    model = DeepLabV3Res101()
    model, device = setup(model)
    print('Loading model from {}.'.format(ckpt))
    model, _ = restore_from(model, ckpt)
    model.eval()

    # 读取并转换图片
    src_img = Image.open(img_path).convert('RGB')
    img = SegDataset([], [], **config.dataset_config).transformX(src_img) # type: torch.Tensor
    img = torch.stack([img], dim=0)
    img.to(device)
    # 执行推理
    y_pred = model(img) # type: torch.Tensor
    y_pred = y_pred.argmax(dim=1).cpu().numpy()[0] # type: np.ndarray
    y_pred_img = np.array(Image.fromarray(y_pred.astype(np.uint8)).resize(src_img.size))

    # 绘制结果图
    src_img = np.array(src_img)
    draw_output(img_path, y_pred_img, src_img)

def draw_output(img_path:str, pred_img:np.ndarray, src_img:np.ndarray):
    '''绘制原始图像和ground truth到一个文件中

    Args:
        img_path(str): 原始图像路径，用来获取标签路径
        pred_img(np.ndarray): 预测数据
        src_img(np.ndarray): 原始图片数据
    '''
    output_path = './output'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    gd_path = img_path.replace('img', 'label')
    gd_array = np.array(Image.open(gd_path))
    print('acc', np.sum(gd_array == pred_img) / gd_array.shape[0] / gd_array.shape[1])

    fig = plt.figure(figsize=(15, 6))
    plot_data = [(src_img, 'Source Image'), (pred_img, 'Predict Image'), (gd_array, 'Ground Truth')]
    for i, (img, title) in enumerate(plot_data):
        ax = fig.add_subplot(131 + i)
        ax.set_title(title)
        if i > 0:
            ax.imshow(img / 3 * 255, cmap='bone')
        else:
            ax.imshow(img)
    output_filename = os.path.join(output_path, os.path.basename(img_path))
    miou, ious, mpa = get_metrics(gd_array, pred_img)

    fig.suptitle('$mIoU={:.2f}, mpa={:.2f}$\n$IoUs={}$'.format(miou, mpa, ious))

    fig.savefig(output_filename)
    print('Output has been saved to {}.'.format(output_filename))

def parse_args():
    parser = argparse.ArgumentParser(usage='python3 eval.py -i path/to/image -r path/to/checkpoint')
    parser.add_argument('-i', '--image', help='path to image')
    parser.add_argument('-r', '--checkpoint', help='path to the checkpoint')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    run_on_single_image(args.image, args.checkpoint)