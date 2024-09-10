#!/bin/bash

# 定义超参数列表
lr_list=(0.1 0.02 0.003)
batch_size_list=(4 8 16 24)

# 循环遍历超参数列表
for lr in "${lr_list[@]}"; do
  for batch_size in "${batch_size_list[@]}"; do
    # 启动任务并放入后台执行
    python your_script.py --lr $lr --batch_size $batch_size &
    
    # 记录后台任务的进程ID
    pid=$!
    
    # 等待任务完成
    wait $pid
  done
done