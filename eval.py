'''
用于存放评估脚本
'''

import torch
from torch import nn
from torch.functional import F
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import os
import argparse
from itertools import product

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
    for (path, tip) in zip([img_path, ckpt], ['图片', '检查点']):
        assert path and os.path.exists(path), '%s不存在: %s' % (tip, path)

    print('Setting up model.')
    model = DeepLabV3Res101()
    model, device = setup(model)
    print('Loading model from {}.'.format(ckpt))
    model, _ = restore_from(model, ckpt)
    model.eval()

    # 读取并转换图片
    src_img = Image.open(img_path).convert('RGB')
    img, _ = SegDataset([], [], 'val', **config.dataset_config).transform_on_eval(src_img) # type: torch.Tensor
    img = torch.stack([img], dim=0)
    img.to(device)
    # 执行推理
    with torch.no_grad():
        y_pred = model(img) # type: torch.Tensor
        y_pred = y_pred.argmax(dim=1).cpu().numpy()[0] # type: np.ndarray
        y_pred_img = np.array(Image.fromarray(y_pred.astype(np.uint8)).resize(src_img.size))

    # 绘制结果图
    draw_output(img_path, {
        'Predict Image': y_pred_img,
        'Source Image': np.array(src_img)
    })

def run_and_refine_single_image(img_path:str, ckpt:str):
    '''对一张大图进行处理并进行多尺度融合

    Args:
        img_path(str): 图片路径
        ckpt(str): 检查点路径
    '''
    for (path, tip) in zip([img_path, ckpt], ['图片', '检查点']):
        assert path and os.path.exists(path), '%s路径不存在: %s' % (tip, path)
    print('Setting up model.')
    model = DeepLabV3Res101()
    model, device = setup(model)
    print('Loading model from {}.'.format(ckpt))
    model, _ = restore_from(model, ckpt)
    model.eval()

    # 获取所有相关图片，读取为PIL对象
    print('Reading images.')
    all_imgs = [img_path] + get_intersect_imgs(img_path) # 图片路径，用于恢复图片尺度信息，首位是原始图像
    imgs = [Image.open(path).convert('RGB') for path in all_imgs]
    img_sizes = [img.size for img in imgs] # 保存size，用于将图片复原

    # 准备转换函数，并转换为tensor
    print('Loading images to device.')
    transform = SegDataset([], [], 'val', **config.dataset_config).transform_on_eval
    imgs = torch.stack([transform(img)[0] for img in imgs], dim=0) # 统一转换为统一长宽的图像
    imgs.to(device)

    # 执行推理，并将推理结果恢复成原尺度图像
    print('Do inferening.')
    with torch.no_grad():
        y_pred = model(imgs) # type: torch.Tensor
        y_pred_arrays = y_pred.argmax(dim=1).cpu().numpy().astype(np.uint8) # type: np.ndarray
        y_pred_imgs = [np.array(Image.fromarray(array).resize(size))
            for size, array in zip(img_sizes, y_pred_arrays)] # 恢复图像尺度

    # 开始拼接，根据目标图像即img_path来确定拼接大小
    scale_index = {size[0]: index for index, size in enumerate(set(img_sizes))} # scale:index映射表
    scale_count = len(scale_index) # 确定有多少个不同的尺度
    final_img = np.zeros((scale_count, ) + img_sizes[0]) # 根据原始图像建立容器，shape=(scale_count, w, h)
    base_img, (bs, bx, by, _) = img_path, extract_info_from_filename(img_path) # 获取基准图像的信息
    for src_img_path, y_pred_img in tqdm(zip(all_imgs, y_pred_imgs), desc='Refining'):
        if src_img_path == base_img:
            # 处理基准图像，bs表示base image scale
            assert y_pred_img.shape == final_img.shape[1:],\
                '图像尺寸不匹配：{}->{}'.format(y_pred_img.shape, final_img.shape[1:])
            final_img[scale_index[bs], :, :] = y_pred_img
        else:
            # 相关图像将进行坐标转换
            es, ex, ey, _ = extract_info_from_filename(src_img_path) # 读入图像尺度信息
            # 确定相交区域的坐标，it_st/it_ed -> (x, y)
            it_st, it_ed = (max(bx, ex), max(by, ey)), (min(bx + bs, ex + es), min(by + bs, ey + es))
            # 坐标分别转换为当前源图像和基准图像数组下标
            cur_area = [(x - ex, y - ey) for x, y in [it_st, it_ed]]
            base_area = [(x - bx, y - by) for x, y in [it_st, it_ed]]
            # 将对应的数据置入最终图像
            final_img[scale_index[es], base_area[0][0]:base_area[1][0], base_area[0][1]:base_area[1][1]]\
                = y_pred_img[cur_area[0][0]:cur_area[1][0], cur_area[0][1]:cur_area[1][1]]
    # 对最终的多尺度图像按像素类别进行多数投票，得到最终refine后的输出图像
    # 需要将数组(scale_count, w, h)转换为(num_classes, w, h)，然后执行argmax即可
    w, h = final_img.shape[1:]
    final_img_with_class = np.zeros((config.num_classes, w, h))
    for s, c in product(range(scale_count), range(config.num_classes)):
        final_img_with_class[c, :, :] += final_img[s, :, :] == c
    final_img = final_img_with_class.argmax(0) # type: np.ndarray
    # 绘制效果图
    draw_output(img_path, {
        'Predict Image': y_pred_imgs[0],
        'Refined Image': final_img,
        'Original Image': np.array(Image.open(img_path).convert('RGB'))
    })

def extract_info_from_filename(filename:str)->(int, int, int, str):
    '''从文件名中获取信息

    Args:
        filename(str): 图像文件名
    
    Return:
        scale(int): 图像大小
        x, y(int, int): 图像起点
        parent_img(str): 原图像文件名
    '''
    filename = os.path.basename(filename) # 防止传入不合法的文件名
    scale, x, y, parent_img = [
        [int, int, int, str][i](s) for i, s in enumerate(filename.split('-'))]
    return scale, x, y, parent_img

def get_intersect_imgs(base_img:str)->list:
    '''提取与base_img相交的所有图片路径

    Args: 
        base_img(str): 目标图像

    Return:
        inter_img_list(list<str>): 与目标图像相交的图像
    '''
    # 提取目标图像信息
    base_img_name = os.path.basename(base_img)
    bs, bx, by, bimg = extract_info_from_filename(base_img_name)
    # 提取目标图像同目录图像
    base_path = base_img.replace(base_img_name, '')
    img_files = os.listdir(base_path)
    # 根据相交关系过滤图像
    filtered_files = []
    for img in img_files:
        filename = os.path.basename(img) # type: str
        es, ex, ey, eimg = extract_info_from_filename(filename)
        if all([filename != base_img_name,
                eimg == bimg,
                max(bx, ex) < min(bx + bs, ex + es), # 判断两个举行是否相交
                max(by, ey) < min(by + bs, ey + es)]):
            filtered_files.append(os.path.join(base_path, filename))
    return filtered_files

def draw_output(img_path:str, imgs:dict):
    '''绘制原始图像和ground truth到一个文件中

    Args:
        img_path(str): 原始图像路径，用来获取标签路径
        imgs(dict<str: np.ndarray>): 输出的图片信息
    '''
    assert 'Predict Image' in imgs, '必须包含预测图像'
    output_path = './output'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    gd_path = img_path.replace('img', 'label')
    gd_array = np.array(Image.open(gd_path))

    # 将Ground Truth加入图像中
    imgs['Ground Truth'] = gd_array

    # 计算一个大致的准确率
    pred_img = imgs['Predict Image']
    print('acc', np.sum(gd_array == pred_img) / gd_array.shape[0] / gd_array.shape[1])

    fig_w, fig_h = 15, int(6 * np.ceil(len(imgs) / 3)) # 宽度固定为15，高为6的整数倍
    fig = plt.figure(figsize=(fig_w, fig_h))
    # 绘制图像
    for i, (title, img) in enumerate(imgs.items()):
        ax = fig.add_subplot(31 + i + np.ceil(len(imgs) / 3) * 100)
        ax.set_title(title)
        if len(img.shape) == 2:
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
    parser.add_argument('-re', '--refine', default=1, help='switch on or not refinement')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.refine == 1:
        run_and_refine_single_image(args.image, args.checkpoint)
    else:
        run_on_single_image(args.image, args.checkpoint)