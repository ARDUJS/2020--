#!/usr/bin/env python
# coding: utf-8

# In[11]:


import config
import time
import torch
from data_process import data_generator
from modeling import classfiy
import logging
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import os


# In[12]:


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# In[13]:


####
# Name: save_best_model
# Function: 在验证集或者训练集上, 保存loss最小或者准确度最高的模型参数。
####
def save_best_model(model, v, data_type='dev', use_accuracy=False, arg={}):
    tip = ""
    for k in arg.keys():
        tip += str(k) + ": " + str(arg[k]) + ", "

    # 保存模型
    if not use_accuracy and data_type == 'dev':
        if config.eval_best_loss > v:
            config.eval_best_loss = v
            state = {'net': model.state_dict()}
            save_path = os.path.join(config.output_dir, config.MODEL_NAME + '_state_dict_' +
                                     data_type + '_loss_' + str(v) + '.model')
            print("Save.......")
            torch.save(state, save_path)
            config.train_best_loss_model = save_path
            config.train_best_loss_model_tip = tip

    # 以精确度作为评估标准
    if use_accuracy and data_type == 'dev':
        if config.eval_best_accuracy < v:
            config.eval_best_accuracy = v
            state = {'net': model.state_dict()}
            save_path = os.path.join(config.output_dir, config.MODEL_NAME + '_state_dict_'
                                     + data_type + '_ac_' + str(v) + '.model')
            print("Save.......")
            torch.save(state, save_path)
            config.train_best_accuracy_model = save_path
            config.train_best_accuracy_model_tip = tip


# In[14]:


####
# Name: model_eval
# Function: 在验证集和测试集上，评估模型
# return: 模型评估结果
####
def model_eval(model, data_loader, data_type='dev'):
    result_sum = {}
    nm_batch = 0
    labels_pred = np.array([])
    labels_true = np.array([])
    for step, batch in enumerate(tqdm(data_loader)):
        batch = tuple(t.to(config.device) for t in batch)
        model.eval()
        with torch.no_grad():
            _, pred = model(batch)
        pred = np.argmax(pred.detach().cpu().numpy(), axis=1)
        labels_pred = np.append(labels_pred, pred)
        true = model.get_labels_data().detach().cpu().numpy()
        labels_true = np.append(labels_true, true)

        result_temp = model.get_result()
        result_sum['loss'] = result_sum.get('loss', 0) + result_temp['loss']
        nm_batch += 1

    result_sum["accuracy"] = accuracy_score(labels_true, labels_pred)
    result_sum["f1"] = f1_score(labels_true, labels_pred, average='macro')
    result_sum["loss"] = result_sum["loss"] / nm_batch
    with open(os.path.join(config.output_dir, config.MODEL_NAME + '_' + data_type + '_result.txt'), 'a+',
              encoding='utf-8') as writer:
        print("***** Eval results in " + data_type + "*****")
        for key in sorted(result_sum.keys()):
            print("%s = %s" % (key, str(result_sum[key])))
            writer.write("%s = %s\n" % (key, str(result_sum[key])))
        writer.write('\n')
    return result_sum


# In[15]:


####
# Name: train
# Function: 训练并评估函数
####
def train(model):
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(config.device, n_gpu))
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model_it_self = model.module if hasattr(model, 'module') else model
    global_step = 0
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_train_steps * 0.1, num_train_steps)
    
    dev_loader = data_generator.get_dev_loader()
    train_loader = data_generator.get_train_loader()
    for epoch in trange(int(config.num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            batch = tuple(t.to(config.device) for t in batch)
            loss, output = model(batch, global_step, -1)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps
            # opt.zero_grad()
            loss.backward()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
            #if global_step % config.print_interval == 0:
            #    print_model_result(model_it_self.get_result())

            if global_step % config.eval_interval == 0 or global_step == num_train_steps:
                if config.do_eval:
                    print("\nepoch:{} global:{}\t".format(epoch, global_step))
                    eval_result = model_eval(model_it_self, dev_loader, data_type='dev')
                    # 保存模型，使用loss为评估标准
#                     print("当前")
                    save_best_model(model_it_self, eval_result['loss'], data_type='dev')
                    if config.SAVE_USE_ACCURACY:
                        save_best_model(model_it_self, eval_result['accuracy'], data_type='dev',
                                        use_accuracy=config.SAVE_USE_ACCURACY)
                    print(config.train_best_accuracy_model_tip)
    shutil.copy(config.train_best_accuracy_model, os.path.join(config.output_dir, 'best_ac_model.bin'))
    shutil.copy(config.train_best_loss_model, os.path.join(config.output_dir, 'best_loss_model.bin'))


# In[16]:


def eval_test(model):
    best_model_path = [os.path.join(config.output_dir, config.eval_best_accuracy_model),
                       os.path.join(config.output_dir, config.eval_best_loss_model)]
    for best_model in best_model_path:
        checkpoint = torch.load(best_model)
        model.load_state_dict(checkpoint['net'], strict=False)
        model = model.to(config.device)
        test_loader = data_generator.get_test_loader()
        print("\n********" + best_model + "********")
        model_eval(model, test_loader, data_type='test')
    pass


# In[17]:


####
# Name: init
# Function: 初始化
####
def init(model):
    if config.init_checkpoint is not None:
        state_dict = torch.load(config.init_checkpoint, map_location='cpu')
        model.bert.load_state_dict(state_dict['net'])


# In[18]:


def main():
    model_set = {
        "Classfy": classfiy,
    }
    start_time = time.time()
    os.makedirs(config.output_dir, exist_ok=True)
    args = len(data_generator.get_labels()), data_generator.get_num_train_steps(), -1
    model = model_set[config.MODEL_NAME](*args)
    model = model.to(config.device)
    init(model)
    if config.do_train:
        train(model)
    if config.do_test:
        eval_test(model)
    end_time = time.time()
    print("总计耗时：%d m" % int((end_time - start_time) / 60))
    pass


if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




