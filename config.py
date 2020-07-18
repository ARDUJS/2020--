#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os

is_pre = False
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 实现卡号匹配
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
do_train = True
# do_train = False

do_test = True
do_eval = True
# do_eval = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_batch_size = 8
test_batch_size = 8
dev_batch_size = 8
gradient_accumulation_steps = 1
num_train_epochs = 3
eval_interval = 1000
print_interval = 1
SAVE_USE_ACCURACY = True
seed = 42
do_lower_case = True
learning_rate = 2e-5
warmup_proportion = 0.1
pretrained = 'bert-base-chinese'
MODEL_NAME = 'Classfy'
init_checkpoint = None


max_seq_length = 256
eval_best_loss = 999
eval_best_accuracy = 0
# eval_best_accuracy_model = '/home/yssong/LW/2020/bert_torch/output_checkpoints_bert_2/best_ac_model_static.bin'
# eval_best_loss_model = '/home/yssong/LW/2020/bert_torch/output_checkpoints_bert_2/best_loss_model_static.bin'
eval_best_accuracy_model = 'best_ac_model.bin'
eval_best_loss_model = 'best_loss_model.bin'
data_dir = 'data/train/train/'
output_dir = '../output_bert_base_0'
if is_pre:
	output_bert = '../output_bert_pre'
# data_dir = '/home/yssong/LW/2020/bert_torch/discourse5_data'
# output_dir = './output_discourse5_data'
local_rank = 0
mlm_probability = 0.15


# In[ ]:




