## 目录说明
mer: micro expression recognization，此目录下是关于微表情识别课题任务相关代码；
所有实验只有训练集和验证集的划分，没有划分test集，因为目前微表情的论文报告均是基于交叉验证方法，也没有指定的测试集进行评估模型性能。


## reference_lib
此文件夹包含与微表情相关的自定义包和它人的包

需要进入此目录下运行
`
pip install -e .
`
进行安装自定义包merlib。


关键的自定义包：
- preprocess: 微表情数据预处理相关包， 可以借鉴其中数据预处理代码，主要包括人脸剪裁、人脸预处理
  - align_face.py 人脸对齐入口文件，算法中采用的方法是openface_align 进行人脸对齐  
  - crop_face.py 人脸剪裁入口文件，算法中采用的是dlib_crop_face_v3进行人脸剪裁


## long_short_action 

环境

- pytorch-lightning==1.9.0，pytorch==1.12，cuda==11.3


基于长短运动进行微表情分类的项目目录。
- configs 一些配置文件
- datatsets 定语一些数据增强等
- lightning_logs 运行时产生的日志、权重文件目录
- models 模型文件 主要文件是long_short_fusenet.py
- one_model_trainer.py 项目入口文件，被scripts中脚本调用
- scripts 包含自定义的脚本文件 主要参考文件是 train_fuse_net_in_all_dataset.py
在此文件中配置 数据集根目录路径、标注文件路径、配置文件路径

并且在database目录中附带了示例数据集路径格式以及标注文件格式。

- 标注文件格式 sub17/EP03_09/96_161 96 132 161 1 :帧序列相对目录 onset帧id apex帧id offset帧id 分类id

1. 运行示例
```
python scripts/train_net_on_various_data.py 'run on the crop_face_two_sides processing method' --use_swin_CA  --datasets smic-hs-e  --num_classes 3   --processing_method='crop_face_two_sides' --fast_dev_run
```
- 'run on the crop_face_two_sides processing method' 是此次项目运行时的备注
- --use_swin_CA使用持续注意力
- --datasets smic-hs-e 指定数据集
- --num_classes 3 指定为3分类
- --processing_method='crop_face_two_sides' 指定预处理的数据集根目录

2. 如运行前通过--fast_dev_run检查
```
python scripts/train_net_on_various_data.py 'run on the crop_face_two_sides processing method' --use_swin_CA  --datasets smic-hs-e  --num_classes 3   --processing_method='crop_face_two_sides' --fast_dev_run
```

