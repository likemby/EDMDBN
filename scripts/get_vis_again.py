import numpy as np
from merlib.helper import plot_confusion,extract_metrics_from_dir
from pathlib import Path

dir = input('Enter your npy path: ')
# dir='/home/mby/computer_vision/mer/basic_ablation/lightning_logs/samm/frames-fusenet-loso/version_4/'
dir=Path(dir)
# np_path=dir/'data4vis.npy'
# loaded_data = np.load(np_path, allow_pickle=True).item()
# len(loaded_data['pred_labels'])
# true_labels,preds_labels =loaded_data['true_labels'], loaded_data['pred_labels']
# labels=['1', '2', '3','4','5']
# clas_report=plot_confusion(true_labels,preds_labels,labels,save_path='confusion.png')

# print(clas_report)


res=extract_metrics_from_dir(dir/'checkpoints')