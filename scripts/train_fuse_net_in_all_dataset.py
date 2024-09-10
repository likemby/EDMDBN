#!/home/mby/miniconda3/envs/pl/bin/python

import os
import subprocess
import argparse


DATASETS = {
    'casme2': {
        3: ('/home/mby/computer_vision/database/CASME2/Cropped_aligned_dlib_crop_more_ROI',
            # '/home/mby/computer_vision/database/CASME2/Cropped_fused',
            '/home/mby/computer_vision/database/CASME2/Cropped_unique_path',
            '/home/mby/computer_vision/database/CASME2/unique_path_data_annotations/3classes-loso-annotations'),
        5: ('/home/mby/computer_vision/database/CASME2/Cropped_aligned_dlib_crop_more_ROI',
            # '/home/mby/computer_vision/database/CASME2/Cropped_fused',
            '/home/mby/computer_vision/database/CASME2/Cropped_unique_path',
            '/home/mby/computer_vision/database/CASME2/unique_path_data_annotations/5classes-loso-annotations'),
    },
    'samm': {
        3: ('/home/mby/computer_vision/database/SAMM/aligned_dlib_crop_more_ROI',
            '/home/mby/computer_vision/database/SAMM/aligned_dlib_crop_face_v3',
             '/home/mby/computer_vision/database/SAMM/unique_path_data_annotations/3classes-loso-annotations'),
        5: ('/home/mby/computer_vision/database/SAMM/aligned_dlib_crop_more_ROI',
            '/home/mby/computer_vision/database/SAMM/aligned_dlib_crop_face_v3',
             '/home/mby/computer_vision/database/SAMM/unique_path_data_annotations/5classes-loso-annotations'),
    },
    'smic-hs-e': {
        3: ('/home/mby/computer_vision/database/SMIC-HS-E/aligned_dlib_crop_more_ROI', 
            '/home/mby/computer_vision/database/SMIC-HS-E/aligned_dlib_crop_twice',
            '/home/mby/computer_vision/database/SMIC-HS-E/unique_path_data_annotations/3classes-loso-annotations'),
    },
    'mmew': {
        3: ('/home/mby/computer_vision/database/MMEW/aligned_dlib_crop_more_ROI',
            '/home/mby/computer_vision/database/MMEW/dlib_crop_twice',
            '/home/mby/computer_vision/database/MMEW/unique_path_data_annotations/3classes-loso-annotations'),
        4: ('/home/mby/computer_vision/database/MMEW/aligned_dlib_crop_more_ROI',
            # '/home/mby/computer_vision/database/MMEW/dlib_crop_twice',
            '/home/mby/computer_vision/database/MMEW/Micro_Expression_unique_path',
            '/home/mby/computer_vision/database/MMEW/unique_path_data_annotations/4classes-loso-annotations'),
        5: ('/home/mby/computer_vision/database/MMEW/aligned_dlib_crop_face_v3',
            '/home/mby/computer_vision/database/MMEW/Micro_Expression_unique_path',
            '/home/mby/computer_vision/database/MMEW/unique_path_data_annotations/5classes-loso-annotations'),
    },
}

CONFIG_FILES = {
    3: "/home/mby/computer_vision/mer/long_short_action/configs/casme2/frames-fusenet-loso.yaml",
    4: "/home/mby/computer_vision/mer/long_short_action/configs/mmew/frames-fusenet-loso-4classes.yaml",
    5: "/home/mby/computer_vision/mer/long_short_action/configs/casme2/frames-fusenet-loso-5classes.yaml",
}


def run_commands(args):
    TEMP_DIR = "temp"
    if not os.path.exists(TEMP_DIR):
        os.mkdir(TEMP_DIR)
    if 'all' in args.datasets:
        args.datasets = list(DATASETS.keys())
    if 0 in args.num_classes:
        args.num_classes = [3,4,5]
    
    for dataset in args.datasets:
        for num_classes in args.num_classes:
            if num_classes not in DATASETS[dataset]:
                continue
            data_root_more_roi,data_root_natural, annotation_path = DATASETS[dataset][num_classes]

            base_config_file = CONFIG_FILES[num_classes]
            log_dir_name = f'frames-fusenet-loso-{num_classes}classes' if num_classes != 3 else 'frames-fusenet-loso'
            purpose = f"fusenet-{num_classes}classes:{args.remark}" 
            this_purpose = f"{dataset}:{purpose}"
            log_file_path = f"{TEMP_DIR}/{this_purpose}-{args.use_natural_face}.log"
            data_root=data_root_natural if args.use_natural_face else data_root_more_roi
            command = generate_command(args,num_classes, dataset,data_root,
                                        annotation_path, purpose, log_dir_name, base_config_file)

            with open(log_file_path, 'w') as f:
                flag=subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT).wait()
            if flag != 0:
                print(f"Error: {this_purpose}")
                exit(1)

def generate_command(args,num_classes,dataset, data_root, annotation_path, purpose, log_dir_name, base_config_file):
    command = [
        "python",
        "one_model_trainer.py",
        f"--purpose={purpose}",
        f"--config_file={base_config_file}",
        f"--data_root={data_root}",
        f"--dataset_name={dataset}",
        f"--log_dir_name={log_dir_name}",
        f"--num_classes={num_classes}",
        f"--batch_size={args.batch_size}",
        f"--annotation_path={annotation_path}",
        "--transform_name=FuseNetTransform",
        "--use_short_action",
        "--use_long_action",
        "--patch_size",
        "8",
        "4",
        "4",
        "--num_workers=14",
        "--val_check_interval=1.0",
    ]
    if not args.debug:
        command.append(f"--purpose={purpose}")
    if args.fast_dev_run:
        command.append("--fast_dev_run=True")
    if args.frames_mode is not None:
        command.append(f"--frames_mode={args.frames_mode}")
    if args.sampling_strategy is not None:
        command.append(f"--sampling_strategy={args.sampling_strategy}")
    if args.block_name is not None:
        command.append(f"--block_name={args.block_name}")
    if args.rgb_diff_abs:
        command.append("--rgb_diff_abs")
    if args.weight_decay is not None:
        command.append(f"--weight_decay={args.weight_decay}")
    if args.use_gap:
        command.append("--use_gap")
    if args.use_feature_concat:
        command.append("--use_feature_concat")
    if args.use_offline_aug:
        command.append("--use_offline_aug")
    if args.use_class_weight:
        command.append("--use_class_weight")
    if args.use_pos_vit:
        command.append("--use_pos_vit")
    if args.use_swin_CA:
        command.append("--use_swin_CA")
    if args.use_feature_conv:
        command.append("--use_feature_conv")
    if args.debug:
        command.append("--debug")
        command.append(f"--all_sample_annotation_path={args.all_sample_annotation_path}")
    return command

def parse_args():
    parser = argparse.ArgumentParser(description="Python version of the bash script",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('remark', type=str, help='First argument: remark')
    parser.add_argument('--datasets', nargs='+',default=['all'], help='Third argument: dataset names (multiple values allowed)', choices=list(DATASETS.keys())+['all'])
    parser.add_argument('--num_classes', nargs='+',default=[0],type=int, help='Fourth argument: num_classes (multiple values allowed)', choices=[3, 4, 5]+[0])
    parser.add_argument('--fast_dev_run', action='store_true', default=False, help='Fifth argument: fast_dev_run (optional)')
    parser.add_argument('--use_offline_aug',default=False,action="store_true",help="启用离线数据增强")
    parser.add_argument('--use_class_weight',action='store_true')
    parser.add_argument('--use_natural_face', action='store_true', default=False, help='Seventh argument: use_natural_face (optional)')
    parser.add_argument('--batch_size', type=int, default=4, help='Sixth argument: batch_size ()')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight_decay (optional)')
    parser.add_argument('--use_pos_vit', action='store_true')
    parser.add_argument('--debug', action='store_true',help='在所有数据上训练，寻找合适的lr和batch_size')
    parser.add_argument('--all_sample_annotation_path', type=str, default=None, help='开启debug模式时，需要指定所有数据的annotation_path')
    
    
    # 对比实验参数
    # [short action]
    parser.add_argument('--sampling_strategy', type=int, default=3, help='Second argument: sampling_strategy', choices=[2, 3])
    parser.add_argument('--use_swin_CA', action='store_true')
    
    # [long action]
    parser.add_argument('--rgb_diff_abs',  action='store_true', default=False, help='abs(rgb_diff)')
    parser.add_argument('--block_name', type=str, default='CABlock',choices=['CBAMBlock', 'BasicBlock', 'CABlock'])
   
    # [Dataset parameters] 0: 前后帧相减，1：后帧减第一帧 2：保持像素帧
    parser.add_argument('--frames_mode',  default=0, type=int,choices= [0, 1, 2])
    parser.add_argument('--use_gap', action='store_true')
    parser.add_argument('--use_feature_concat', action='store_true')
    parser.add_argument('--use_feature_conv', action='store_true')


    args=parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()
    run_commands(args)


if __name__ == "__main__":
    main()
