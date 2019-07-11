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
from pyvips import Image as vipImage

import os
import argparse
from itertools import product

from model import DeepLabV3Res101
from utils import setup, restore_from, get_metrics
from dataloader import SegDataset
import config

def run_on_single_image(img_path:str, ckpt:str, model:nn.Module = None, device:int = None)->np.ndarray:
    '''对单张图片进行处理

    Args:
        img_path(str): 图片路径
        ckpt(str): 检查点路径
    '''
    for (path, tip) in zip([img_path, ckpt], ['图片', '检查点']):
        assert path and os.path.exists(path), '%s不存在: %s' % (tip, path)

    if model is None and device is None:
        model, device = setup(DeepLabV3Res101())
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

    # # 绘制结果图
    # draw_output(img_path, {
    #     'Predict Image': y_pred_img,
    #     'Source Image': np.array(src_img)
    # })
    return y_pred_img

def run_and_refine_single_image(img_path:str, ckpt:str, show_output:bool = True,
        model:nn.Module = None, device:int = None)->np.ndarray:
    '''对一张大图进行处理并进行多尺度融合

    Args:
        img_path(str): 图片路径
        ckpt(str): 检查点路径
    '''
    def print_(s):
        if show_output:
            print(s)

    for (path, tip) in zip([img_path, ckpt], ['图片', '检查点']):
        assert path and os.path.exists(path), '%s路径不存在: %s' % (tip, path)
    if model is None and device is None:
        print_('Setting up model.')
        model, device = setup(DeepLabV3Res101())
        print_('Loading model from {}.'.format(ckpt))
        model, _ = restore_from(model, ckpt)
        model.eval()

    # 获取所有相关图片，读取为PIL对象
    print_('Reading images.')
    all_imgs = [img_path] + get_intersect_imgs(img_path) # 图片路径，用于恢复图片尺度信息，首位是原始图像
    imgs = [Image.open(path).convert('RGB') for path in all_imgs]
    img_sizes = [img.size for img in imgs] # 保存size，用于将图片复原，注意：size->(w, h)

    # 准备转换函数，并转换为tensor
    print_('Loading images to device.')
    transform = SegDataset([], [], 'val', **config.dataset_config).transform_on_eval
    imgs = torch.stack([transform(img)[0] for img in imgs], dim=0) # 统一转换为统一长宽的图像
    imgs.to(device)

    # 执行推理，并将推理结果恢复成原尺度图像
    print_('Do inferening.')
    with torch.no_grad():
        y_pred = model(imgs) # type: torch.Tensor
        y_pred_arrays = y_pred.argmax(dim=1).cpu().numpy().astype(np.uint8) # type: np.ndarray
        y_pred_imgs = [np.array(Image.fromarray(array).resize(size))
            for size, array in zip(img_sizes, y_pred_arrays)] # 恢复图像尺度

    # 统计总共有多少尺度
    scales = set()
    for p in all_imgs:
        bs, _, _, _ = extract_info_from_filename(p)
        scales.add(bs)
    # 开始拼接，根据目标图像即img_path来确定拼接大小
    scale_index = {scale: index for index, scale in enumerate(scales)} # scale:index映射表
    scale_count = len(scales) # 确定有多少个不同的尺度
    fw, fh = img_sizes[0] # 在边界时原始图像大小可能不等于scale*scale，所以要以原始图像大小为准
    final_img = np.zeros((scale_count, fh, fw)) # 根据原始图像建立容器，shape=(scale_count, h, w)
    base_img, (bs, bx, by, _) = img_path, extract_info_from_filename(img_path) # 获取基准图像的信息
    for src_img_path, y_pred_img in tqdm(zip(all_imgs, y_pred_imgs), desc='Refining'):
        if src_img_path == base_img:
            # 处理基准图像，bs表示base image scale
            if y_pred_img.shape != final_img.shape[1:]:
                print('图像尺寸不匹配：{}->{}'.format(y_pred_img.shape, final_img.shape[1:]))
            final_img[scale_index[bs], :, :] = y_pred_img[0:fh, 0:fw]
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
    }, show_output=show_output)
    # 返回refine过的图像
    return final_img

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

def draw_output(img_path:str, imgs:dict, show_output:bool = True):
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
    gd_array = np.array(Image.open(gd_path)) if os.path.exists(gd_path) else None

    # 将Ground Truth加入图像中
    if gd_array is not None:
        imgs['Ground Truth'] = gd_array
        pred_img = imgs['Predict Image']

    # 绘制图例
    legend_array = np.zeros((100, 800), dtype=np.uint8)
    for i in range(config.num_classes):
        legend_array[:, i * 200:(i + 1) * 200] = i
    legend_array = legend_array / (config.num_classes - 1) * 255
    imgs['Legend'] = legend_array

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
    # 计算各项指标
    if gd_array is not None:
        miou, ious, acc = get_metrics(gd_array, pred_img)
        fig.suptitle('$mIoU={:.2f}, acc={:.2f}$\n$IoUs={}$'.format(miou, acc, ['%.2f' % x for x in ious]))
    # 获取原始文件名，并根据文件名得到输出目录信息
    filename = os.path.basename(img_path)
    _, _, _, parent_img = extract_info_from_filename(filename)
    output_path = os.path.join(output_path, parent_img.replace('.png', '')) # 按父文件名分类
    output_filename = os.path.join(output_path, filename)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    fig.savefig(output_filename)
    if show_output:
        print('Output has been saved to {}.'.format(output_filename))

def run_on_large_image(root_path:str, original_img:str, ckpt:str):
    '''在一整张图上执行运算

    Args:
        root_path(str): 从大图中提取的小图所在目录
        original_img(str): 大图的路径
        ckpt(str): 检查点路径
    '''
    assert os.path.exists(root_path), '路径不存在: {}'.format(root_path)
    assert os.path.exists(original_img), '路径不存在: {}'.format(original_img)
    # 配置模型
    print('Setting up model.')
    model, device = setup(DeepLabV3Res101())
    print('Loading model from {}.'.format(ckpt))
    model, _ = restore_from(model, ckpt)
    model.eval()

    file_list = os.listdir(root_path)
    original_img_array = vipImage.new_from_file(original_img)
    ow, oh = [getattr(original_img_array, key) for key in ['width', 'height']]
    large_array = np.zeros((oh, ow), dtype=np.uint8)
    # 统计尺度
    scales = set()
    for file in file_list:
        scale, _, _, _ = extract_info_from_filename(file)
        scales.add(scale)
    # 使用最大尺度的图片作为索引
    # max_scale = max(scales)
    max_scale = 600 # 使用600作为索引
    for file in tqdm(file_list, desc='Processing'):
        scale, x, y, _ = extract_info_from_filename(file)
        if scale != max_scale: continue
        file_path = os.path.join(root_path, file)
        # refined_image = run_and_refine_single_image(file_path, ckpt, False, model=model, device=device)
        refined_image = run_on_single_image(file_path, ckpt, model=model, device=device)
        rh, rw = refined_image.shape # 根据生成图像大小填充最终图像
        large_array[x:x + rh, y:y + rw] = refined_image
    print('Saving to files...')
    # 为输出图片上色，方便查看
    colored_array = large_array * 85
    # 输出目录
    filename, extension = os.path.basename(original_img).rsplit('.', 1)
    output_file = os.path.join('./output', '{}_prediction.{}'.format(filename, extension))
    colored_file = os.path.join('./output', '{}_prediction_colored.{}'.format(filename, extension))
    vipImage.new_from_memory(large_array.data, ow, oh, 1, 'uchar').write_to_file(output_file)
    vipImage.new_from_memory(colored_array.data, ow, oh, 1, 'uchar').write_to_file(colored_file)
    print('Large Image File has been saved to {} and {}'.format(output_file, colored_file))

def parse_args():
    parser = argparse.ArgumentParser(usage='python3 eval.py -i path/to/image -r path/to/checkpoint')
    parser.add_argument('-i', '--image', help='path to image')
    parser.add_argument('-r', '--checkpoint', help='path to the checkpoint')
    parser.add_argument('-re', '--refine', default=1, help='switch on or not refinement')
    parser.add_argument('-ro', '--root_path', help='root path of images')
    parser.add_argument('-oi', '--original_img', help='source image')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.root_path:
        run_on_large_image(args.root_path, args.original_img, args.checkpoint)
    elif args.refine == 1:
        run_and_refine_single_image(args.image, args.checkpoint)
    else:
        run_on_single_image(args.image, args.checkpoint)
