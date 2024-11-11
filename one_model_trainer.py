from typing import Any, List, Dict, Optional, Union
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import nn, optim
from torchmetrics import Accuracy, F1Score, Recall, MetricCollection
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from timm.models import create_model
import torch
import torch.nn.functional as F
import yaml
# import timm.optim.optim_factory as optim_factory
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
# import pytorchvideo.models as pvm
import sys
# import models.mmnet
import models.long_short_fusenet
from datasets.label_datasets import add_dataset_specific_args, get_kfolddatamodules_from_args,get_all_data
from merlib.helper import extract_metrics, send_email,get_print_logger
from merlib.models.utils.focal_loss import focal_loss
import merlib.models.utils.weight_lr_decay as wld
import merlib.models.utils.lr_sched as lr_sched
from merlib.data import TEMP_DIR_PATH

torch.set_float32_matmul_precision('medium')

class LitModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.p_logger=get_print_logger(self.hparams.debug)

        self.model = self._build_model()

        num_classes = self.hparams.num_classes
        self.val_metric = MetricCollection({
            'acc': Accuracy(num_classes=num_classes),
            'uf1': F1Score( num_classes=num_classes, average='macro'),
            'uar': Recall(num_classes=num_classes, average='macro')
        })
        self.vis_flag=False
        self.the_best_test_result =None
        self.pred_labels = []
        self.true_labels = []
        self.feature_outputs=[]


    def forward(self, x):
        # only used for inference or predict
        return self.model(x).argmax(dim=1)

    def _build_model(self) -> nn.Module:
        num_frames = self.hparams.num_frames
        # drop_path = self.hparams.drop_path
        img_size = self.hparams.input_size
        pretrained = self.hparams.use_2dpretrained  # type: ignore
        print(f'使用预训练模型参数：{bool(pretrained)}')
        
        model = create_model(
            self.hparams.model, pretrained=bool(pretrained), **self.hparams)
        return model


    def calculate(self, inputs, labels):
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)  # (max values,max indexes)
        if args.use_class_weight:
            class_weight = torch.tensor(
                self.hparams.blanced_weights, dtype=torch.float, device=labels[0].device)
        else:
            class_weight = None

        if self.hparams.cls_loss == 'focal':
            cls_loss = focal_loss(
                alpha=class_weight, gamma=1, device=labels[0].device)(outputs, labels)
        elif self.hparams.cls_loss == 'ce':
            cls_loss = F.cross_entropy(
                outputs, labels, label_smoothing=0.1, weight=class_weight)
        else:
            assert False, 'no this cls loss'
        loss = cls_loss
        return loss, preds

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        # forward
        loss, preds = self.calculate(inputs, labels)
        if not torch.isfinite(loss):
            self.p_logger.error("Loss is {}, stopping training".format(loss))
            sys.exit(1)
        self.log('loss',loss, on_step=False,
                 on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        loss, preds = self.calculate(inputs, labels)
        self.log('val_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=False)
        step_output={'preds': preds, 'labels': labels}
        if self.vis_flag:
            feature_vector = self.model.forward_features(inputs).cpu()
            step_output['features']=feature_vector.numpy()
        return  step_output # 适应数据并行

    def validation_epoch_end(self, outputs) -> None:
        preds, labels = [x['preds']
                         for x in outputs],  [x['labels'] for x in outputs]
        metric_dict = self.val_metric(torch.hstack(preds).flatten(),
                                      torch.hstack(labels).flatten())
        self.val_metric.reset()
        
        self.log_dict(metric_dict, on_step=False,
                      on_epoch=True, prog_bar=True)

        if self.vis_flag:
            print('on---------------------vis_flag')
            # 为聚类分析做准备
            [self.feature_outputs.extend(x['features']) for x in outputs]
            # 为计算多折混淆矩阵准备
            preds = torch.hstack(preds).flatten().cpu()
            labels = torch.hstack(labels).flatten().cpu()

            num_classes = self.hparams.num_classes
            val_metric = MetricCollection({
            'acc': Accuracy(num_classes=num_classes),
            'uf1': F1Score( num_classes=num_classes, average='macro'),
            'uar': Recall(num_classes=num_classes, average='macro')
        })
            self.the_best_test_result=val_metric(preds, labels)
            self.pred_labels = preds
            self.true_labels = labels


    def _get_eff_batch_size(self):
        batch_size = self.hparams.batch_size
        accumulate_grad_batches = self.hparams.accumulate_grad_batches
        num_devices = self.trainer.num_devices
        num_node = self.trainer.num_nodes

        eff_batch_size = batch_size * accumulate_grad_batches * num_devices*num_node
        print('eff_batch_size: ', eff_batch_size)
        return eff_batch_size

    def configure_optimizers(self):

        # linear learning rate
        eff_bs = self._get_eff_batch_size()
        lr = self.hparams.lr
        if lr is None:  # only base_lr is specified
            lr = self.hparams.blr * eff_bs / 256
        print('real initial learning rate: ', lr)

        # optimizer
        weight_decay = self.hparams.weight_decay
        layer_decay = self.hparams.layer_decay
        opt_betas = self.hparams.opt_betas

        # build optimizer with layer-wise lr decay (lrd)
        if layer_decay!=1.0:
            print('use_lr_decay:', layer_decay)
            self.model.set_lr_decay(layer_decay,all_the_same=True)

        param_groups = wld.build_param_groups(
            self.model, weight_decay,use_lr_scale=True)

        optimizer = torch.optim.AdamW(
           param_groups, lr=lr, betas=opt_betas, capturable=True)

        # scheduler
        total_steps = self.trainer.estimated_stepping_batches
        total_epochs = self.trainer.max_epochs
        every_epoch_steps = total_steps//total_epochs
        warmup_epochs = self.hparams.warmup_epochs

        lr_scheduler = lr_sched.CosineWarmupScheduler(optimizer, warmup=warmup_epochs*every_epoch_steps,
                                                      max_iters=total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step"
            }
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MER Model")
        # [Model parameters]
        parser.add_argument('--model', default='mae3d_vit_base_patch16_dec512d4b', type=str, metavar='MODEL',
                            help='Name of model to train')
        parser.add_argument('--num_classes', type=int, default=3)
        parser.add_argument('--use_long_action', action='store_true')
        parser.add_argument('--use_short_action', action='store_true')
        parser.add_argument('--use_feature_concat', action='store_true')
        parser.add_argument('--use_feature_conv', action='store_true')
        parser.add_argument('--use_pos_vit', action='store_true')

        parser.add_argument('--use_gap', action='store_true')
        parser.add_argument('--use_gmp', action='store_true')

        parser.add_argument('--block_name', type=str, default='CBAMBlock',choices=['CBAMBlock', 'BasicBlock', 'CABlock'])
        # in video Swin: 0.1,0.2,0.3 for Swin-T,Swin-S,Swin-B
        # in 2d Swin-T,0.2, 0.3, 0.5 for Swin-T, Swin-S, and Swin-B, 0.1 for finetuning
        parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                            help='Drop path rate (default: 0.1)')
        parser.add_argument('--window_size', type=int, default=[8, 7, 7], nargs='+',
                            help='Swin Window size')
        parser.add_argument('--patch_size', type=int, default=[2, 4, 4], nargs='+',
                            help='Swin patch size')
        parser.add_argument('--use_swin_CA', action='store_true')
        parser.add_argument(
            '--cls_head', type=str, choices=['vit_gap', 'vit_cls', 'resnet_gap','linear'], default='vit_cls')

        # [Dataset parameters] 0: 前后帧相减，1：后帧减第一帧 2：保持像素帧
        parser.add_argument('--frames_mode',  default=0, type=int,choices= [0, 1, 2])
        parser.add_argument('--rgb_diff_abs',  action='store_true', default=False, help='abs(rgb_diff)')
        # imagenet 21k pretrained file
        parser.add_argument('--use_2dpretrained', default=False, action="store_true",
                            help='use imagenet 21k pretrained checkpoint')

        # [loss]
        parser.add_argument('--cls_loss', default='focal', choices=['focal', 'ce'], type=str, metavar='LOSS',
                            help='Name of loss to train')
        # [optimizer]
        # in 2d swin-T, 1e-3 cosine schduler for  in1k and in21k pretrain,  1e-5 constant lr for finetuning
        parser.add_argument('--lr', type=float, default=None, metavar='LR',
                            help='learning rate (default: 1.6e-3)')
        parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                            help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
        parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                            help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
        # in 2d swin, around max_epochs * 1/15
        parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                            help='epochs to warmup LR, if scheduler supports')
        # in video Swin: 0.02,0.02,0.05 for Swin-T,Swin-S,Swin-B
        # in 2d swin,maybe 1e-8 for finetuning
        parser.add_argument('--weight_decay', type=float, default=0.05,
                            help='weight decay (default: 0.05)')
        parser.add_argument('--layer_decay', type=float, default=0.75,
                            help='layer-wise lr decay from ELECTRA/BEiT')
        # in 2d Swin, opt_betas from scratch by [0.9, 0.95]; finetune by [0.9, 0.999]
        parser.add_argument('--opt_betas', type=float, default=[0.9, 0.95], nargs='+', metavar='BETA',
                            help='Optimizer Betas (default: None, use opt default)')
        return parent_parser


def get_args(*, show=True):
    import argparse
    import sys
    from argparse import Namespace
    from merlib.helper import print_dict_as_table
    parser = argparse.ArgumentParser(
        '3D Micro Expression Recognization --- Basic ablation experiments!', add_help=False)
    # 记录执行的参数命令
    parser.set_defaults(args_str=" ".join(sys.argv))
    # log_dir_name
    parser.add_argument('--log_dir_name', default=None, type=str)

    # describe the purpose and status of this experiment
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--purpose', default='debug',
                       help="describe the purpose of the experiment")
    group.add_argument('--debug', action='store_true', default=False,
                       help="help describe the purpose of the experiment and indicate status")

    # config file
    parser.add_argument(
        '--config_file', default="指定的配置文件")
    temp_args, unexpected_params_list = parser.parse_known_args()
    assert Path(temp_args.config_file).exists(),temp_args.config_file
    yaml_dict = yaml.safe_load(yaml_path.open()) if (
        yaml_path := Path(temp_args.config_file)).exists() else {}

    # global random seed
    parser.add_argument('--seed', default=42, type=int)

    # resume checkpoint
    parser.add_argument('--resume_ckpt_path', default=None,
                        type=str, help='resume from checkpoint')
    parser.add_argument('--resume_fold', default=0,
                        type=int, help='resume from specific fold')
    # 如果不设置此项，讲开启另一个version，不利于计算度量指标
    parser.add_argument('--resume_logger_vision', default=None,
                        type=int, help='resume from specific logger version')

    # dataset args
    parser = add_dataset_specific_args(parser)
    # model args
    parser = LitModel.add_model_specific_args(parser)
    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    # args = parser.parse_args()

    # 优先级：命令行传参 > yaml配置文件 > 参数默认值
    args = parser.parse_args(namespace=Namespace(**yaml_dict))

    # 打印传递的参数，方便检验
    if show:
        print_dict_as_table(args.__dict__)
    return args


def main(args):
    # workers=True ensures that data augmentations are not repeated across workers.
    pl.seed_everything(args.seed, workers=True)

    # logger
    p_logger=get_print_logger(args.debug)
    logger_version = args.resume_logger_vision if args.resume_logger_vision is not None else None
    logger_save_dir = Path(args.default_root_dir) / \
        'lightning_logs'/args.dataset_name
    log_dir_name = ('debug' if args.debug else args.log_dir_name)

    Global_V = {
        'true_labels': [],
        'pred_labels': [],
        'labels': [],
        'feature_vectors':[]
    }
    
    if args.debug:
        all_sample_datamodule=get_all_data(args)
        args.blanced_weights = all_sample_datamodule.weights
        logger = TensorBoardLogger(
            save_dir=logger_save_dir,  # type: ignore
            name=log_dir_name,
            version=logger_version,
            sub_dir='Training'
        )
        checkpoint_callback = ModelCheckpoint(
            filename='{epoch}-{acc:.4f}-{uf1:.4f}-{uar:.4f}',
            save_top_k=-1,
            # save_last=True,
            # monitor='uf1',
            # mode="max",
            every_n_epochs=10,
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = pl.Trainer.from_argparse_args(args,callbacks=[lr_monitor,checkpoint_callback],logger=logger,
                                                auto_lr_find=True,
                                                )
        model = LitModel(**vars(args))
        # tune_res=trainer.tune(model,all_sample_datamodule)
        # p_logger.debug(f'---------------------------log_dir: {trainer.log_dir}---------------------------')
        # lr_finder=tune_res['lr_find']
        # fig = lr_finder.plot(suggest=True)
        # fig.savefig(Path(trainer.log_dir)/'lr_finder.png')
        # Pick point based on plot, or get suggestion
        # new_lr = lr_finder.suggestion()
        # p_logger.debug(f'---------------------------new_lr: {new_lr}---------------------------')
        trainer.fit(model, datamodule=all_sample_datamodule)
        logger_version = logger.version
        # args.lr=new_lr
        return True
    kfold_datamodules = get_kfolddatamodules_from_args(args, args.resume_fold)
    for fold, datamodule in enumerate(kfold_datamodules, start=args.resume_fold):
        if args.fast_dev_run and fold > 0:
            break
        # train model
        p_logger.info('training at fold: {}'.format(fold))
        args.blanced_weights = datamodule.weights
        checkpoint_callback = ModelCheckpoint(
            filename="fold{}-".format(fold) +'{epoch}-{acc:.4f}-{uf1:.4f}-{uar:.4f}',
            save_top_k=1,
            save_last=False,
            monitor='uf1',
            mode="max",
        )

        logger = TensorBoardLogger(
            save_dir=logger_save_dir,  # type: ignore
            name=log_dir_name,
            version=logger_version,
            sub_dir='folds/fold{}'.format(fold)
        )
        if logger_version is None:
            logger_version = logger.version

        lr_monitor = LearningRateMonitor(logging_interval='step')
        patience=args.max_epochs *0.5 //args.val_check_interval
        early_stopping = EarlyStopping(
            monitor='uf1', mode="max", stopping_threshold=0.99, patience=patience)

        trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback, lr_monitor, early_stopping],
                                                logger=logger,
                                                enable_model_summary=(
                                                    fold == args.resume_fold)
                                                )

        model = LitModel(**vars(args))
        resume_ckpt_path = args.resume_ckpt_path if fold == args.resume_fold else None
        trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt_path)
        
        if not args.fast_dev_run:
            model.vis_flag=True
            trainer.validate(model,ckpt_path='best', datamodule=datamodule)
            
            Global_V['pred_labels'].extend(model.pred_labels)
            Global_V['true_labels'].extend(model.true_labels)
            Global_V['labels'] = datamodule.labels
            Global_V['feature_vectors'].extend(model.feature_outputs)
    else:
        save_log_dir_path= Path(trainer.log_dir).parent.parent

    if args.fast_dev_run:
        return
    # 保存数据
    import numpy as np
    np.save(save_log_dir_path/'data4vis.npy', Global_V, allow_pickle=True)

    try:
        # 保存聚类分析图片
        from merlib.helper import plot_clustering
        save_img_path = save_log_dir_path/'clustering.png'
        plot_clustering( Global_V['feature_vectors'],args.num_classes,save_path=save_img_path)
        plot_clustering(Global_V['feature_vectors'], args.num_classes, save_path=save_img_path)
    except ValueError as e:
        # 处理值错误异常
        print("发生值错误异常:", str(e))

    # 保存混淆矩阵图片
    from merlib.helper import plot_confusion
    true_labels = torch.tensor(Global_V['true_labels']).numpy()
    preds_labels = torch.tensor(Global_V['pred_labels']).numpy()
    labels = Global_V['labels']
    save_img_path = save_log_dir_path/'confusion_matrix.png'
    cls_report=plot_confusion(true_labels, preds_labels, labels, save_path=save_img_path)

    # 保存模型权重
    # 解析checkpoints目录下权重文件，生成结果
    checkpoints_dir: Path = checkpoint_callback.dirpath
    re_path = 'fold(?P<fold>\d+)-epoch=(?P<epoch>\d+)-acc=(?P<acc>[.0-9]{6})-uf1=(?P<uf1>[.0-9]{6})-uar=(?P<uar>[.0-9]{6})'
    save_file_path = save_log_dir_path/"{}.txt".format(args.purpose)
    result_summary = extract_metrics(re_path, Path(
        checkpoints_dir).glob('*.ckpt'), save_file_path,extra_words=cls_report)

    # 结束训练，发送邮件通知
    send_email(
        addr_from='xxxxxxxxxx',  # 发送方邮箱
        authorization_code="xxxxxx",
        addr_to='xxxxxx',  # 接收方邮箱
        smtp_server='xxxxx',
        head_from='我的程序',
        head_to='本人',
        head_subject='程序结束提醒',
        message='程序已经跑完,相关结果保存在\n{}\n{}\n{}'.format(save_file_path, result_summary, cls_report)
    )


if __name__ == '__main__':
    args = get_args(show=True)
    if args.default_root_dir:
        Path(args.default_root_dir).mkdir(parents=True, exist_ok=True)
    main(args)
