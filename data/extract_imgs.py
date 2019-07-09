'''
数据处理函数
'''
import os
from itertools import product
from argparse import ArgumentParser

from pyvips import Image as vipImage
import numpy as np
from cv2 import cv2
from tqdm import tqdm

def mk_folder(path:str):
    '''创建目录

    Args:
        path(str): 目标路径
    '''
    if not os.path.exists(path):
        os.mkdir(path)
        print('{} created!'.format(path))

def create_folders(path:str):
    '''生成对应的目录

    Args:
        path(str): 目标路径

    Return:
        此函数在$path路径下生成img和label两个路径
    '''
    mk_folder(path)
    
    for folder in ['img', 'label']:
        full_path = os.path.join(path, folder)
        mk_folder(full_path)

def load_to_memory(img)->np.ndarray:
    '''将图片完全加载到内存，低于32GB内存的机器慎用

    Args:
        img: pyvips image对象

    Return:
        img_arr: numpy数组
    '''
    format_to_dtype = {
        'uchar': np.uint8,
        'char': np.int8,
        'ushort': np.uint16,
        'short': np.int16,
        'uint': np.uint32,
        'int': np.int32,
        'float': np.float32,
        'double': np.float64,
        'complex': np.complex64,
        'dpcomplex': np.complex128,
    }
    return np.ndarray(
        buffer=img.write_to_memory(),
        dtype=format_to_dtype[img.format],
        shape=[img.height, img.width, img.bands]
    )

def is_img_empty(img_array:np.ndarray, threshold:float)->bool:
    '''判断图片是否为空

    Args:
        img_array(np.ndarray): 表示图像的numpy数组
        threshold(float): 空白区域占比，大于这个比例，将返回False

    Return:
        is_empty(bool): 是否为空
    '''
    if not (len(img_array.shape) == 3 and img_array.shape[-1] == 4):
        raise ValueError('img_array应该为四通道图像')
    
    w, h = img_array.shape[0:2]
    opacity_mask = img_array[:, :, 3] # 抽取透明度通道
    opacity_ratio = np.sum([opacity_mask == 0]) / w / h # 统计透明像素占比
    return opacity_ratio >= threshold

def extract_imgs(img_path:str, label_path:str, output_path:str):
    '''读取图像并进行切分处理

    Args:
        img_path(str): 要处理的图像路径
        label_path(str): 对应的标签图像
        output_path(str): 输出目录
    '''
    print('Loading source image from {}.'.format(img_path))
    src_img = vipImage.new_from_file(img_path)
    
    label_img = None
    if label_path is not None:
        print('Loading label image from {}.'.format(label_path))
        label_img = vipImage.new_from_file(label_path)

    # 输出图片基本信息
    for img in [('Source Image', src_img), ('Label Image', label_img)]:
        print('{} info:'.format(img[0]))
        if img[1] is None: continue
        for key in ['width', 'height', 'bands']:
            print('\t{}: {}'.format(key, getattr(img[1], key)))

    # 将图像完全加载到内存，耗时可能会很长，内存小的机器慎用
    print('Loading image to memory, it may take a few minutes.')
    src_img = load_to_memory(src_img)
    label_img = load_to_memory(label_img) if label_img else None
    crop_and_save(src_img, label_img, output_path, img_path)

def crop_and_save(src_img:np.ndarray, label_img:np.ndarray, output_path:str, img_path:str,
        need_std_mean:bool = False):
    '''裁剪图片并保存
    Args:
        src_img, label_img(np.ndarray): 原始图片数组和标签数组
        output_path(str): 输出目录
        img_path(str): 原始图像路径
    '''
    # 获取要处理的图片文件名
    img_file_name = os.path.basename(img_path).rsplit('.')[0]
    # 获取输出目录的路径
    src_output_path = os.path.join(output_path, 'img')
    label_output_path = os.path.join(output_path, 'label')

    # 开始裁剪图片
    scales = [480, 600, 960, 1280] # 多尺度裁切
    build_iter = lambda n, s: range(0, (n // s + 1) * s, s) # 范围覆盖(0, n)且步长为s的迭代器

    # 统计图像总数目
    img_count = 0
   
    for scale in scales:
        # 这里会根据实际图像大小进行裁剪
        [h, w] = src_img.shape[0:2]
        # 使用两个迭代器完成裁切工作
        width_iter = build_iter(w, scale)
        height_iter = build_iter(h, scale)
        w_h_iter = product(height_iter, width_iter) # product生成两个迭代器的笛卡尔积
        total = len(width_iter) * len(height_iter) # 裁剪后图像总数量
        imgs = []
        for (x, y) in tqdm(w_h_iter, desc='Scale[{}*{}]'.format(scale, scale), total=total):
            # 执行裁剪
            cropped_src_img = src_img[x:x + scale, y:y + scale, :] # 只抽取RGB通道
            cropped_label_img = label_img[x:x + scale, y:y + scale, :] if label_img else None
            # 过滤掉空白区域
            if not is_img_empty(cropped_src_img, 0.95):
                img_count += 1
                # 收集所有的像素值，用于计算均值和方差
                w, h, c = cropped_src_img.shape
                reshaped_crop = cropped_src_img.reshape((w * h, c))[:, 0:3]
                imgs.append(reshaped_crop)
                # 写入硬盘
                filename = '{}-{}-{}-{}.png'.format(scale, x, y, img_file_name)
                # 只需写入RGB值
                cv2.imwrite(os.path.join(src_output_path, filename), cropped_src_img[:, :, 0:3])
                if cropped_label_img:
                    cv2.imwrite(os.path.join(label_output_path, filename), cropped_label_img)
        if need_std_mean:
            # 计算数据集的均值和方差
            print('Concating images.')
            imgs = np.concatenate(imgs)
            mean = list(np.mean(imgs, axis=0))
            std = list(np.std(imgs, axis=0))
            print('Mean: {}, Std: {}'.format(mean, std))
    print('Extracted {} images from {}'.format(img_count, img_path))

def parse_args():
    '''解析命令行参数'''
    arg_parser = ArgumentParser(usage='python3 extract_imgs.py -i path/to/large/img -l path/to/label/img -o path/to/output')
    arg_parser.add_argument('-i', '--img', help='卫星图路径')
    arg_parser.add_argument('-l', '--label', help='卫星图标签图路径')
    arg_parser.add_argument('-o', '--output', help='输出路径')

    return arg_parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    assert os.path.exists(args.img), '请指定有效的图片路径'
    if args.label:
        assert os.path.exists(args.label), '请指定有效的标签图片路径'
    assert args.output is not None, '请指定有效果的输出路径'
    create_folders(args.output)
    extract_imgs(args.img, args.label, args.output)