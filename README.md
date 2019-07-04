# Ali-Agriculture-AI-Challenge-2019
阿里天池2019年县域农业大脑AI挑战赛

## 依赖库
```
1. pyvips
sudo apt install libvips
pip install pyvips

2. opencv
pip install opencv-python
```

## 数据预处理
1. 将RGBA图片处理为RGB图片，其中透明度为0的区域像素值设置为`(0, 0, 0)`，其他则直接提取出RGB通道。
2. 数据增强，包括多尺度裁剪、随机裁剪、多角度旋转（0/15/30/45/60/75/90）、镜像、翻转等。
3. 减去均值，做归一化。

## 网络设置
采取`pytorch`内置的`deeplabv3_resnet101`网络，由于PASCAL数据集与卫星云图数据集相差较远，所以不采用预训练权重。

## 训练策略
[TODO]