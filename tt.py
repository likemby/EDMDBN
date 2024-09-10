import time
import torch
from torchvision.models import resnet18
from torch.backends import cudnn
import tqdm
import numpy as np
import pandas as pd
from one_model_trainer import LitModel
cudnn.benchmark = True

# 1.构造模型
check_point='/home/mby/computer_vision/mer/long_short_action/lightning_logs/casme2/frames-short_action-loso-5classes/version_11/checkpoints/fold0-epoch=2-acc=0.5882-uf1=0.5248-uar=0.6889.ckpt'
model = LitModel.load_from_checkpoint(check_point)
model.eval()
