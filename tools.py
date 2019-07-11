import numpy as np
from pyvips import Image

import os
import argparse

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

def add_color(img:str):
    '''为图片上色
    '''
    assert os.path.exists(img), '路径不存在: {}'.format(img)
    large_img = Image.new_from_file(img)
    print('Loading to memory...')
    large_array = load_to_memory(large_img)
    large_array = large_array * 80
    print(large_array.shape)
    h, w, c = large_array.shape
    output_file = os.path.basename(img) # type: str
    filename, extension = output_file.rsplit('.', 1)
    output_file = '{}_colored.{}'.format(filename, extension)
    Image.new_from_memory(large_array.data, w, h, c, large_img.format)\
        .write_to_file(output_file)
    print('Write to {}'.format(output_file))

def thumbnail(img:str):
    '''生成图片缩略图
    '''
    scale = 0.2
    filename, extension = os.path.basename(img).rsplit('.', 1)
    output_file = os.path.join('./', '{}_{}.{}'.format(filename, scale, extension))
    Image.new_from_file(img).resize(scale).write_to_file(output_file)
    print('Write to {}'.format(output_file))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help='path to image')
    parser.add_argument('-t', '--type', default='color')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.type == 'color':
        add_color(args.image)
    else:
        thumbnail(args.image)