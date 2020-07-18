#!/usr/bin/env python
# coding: utf-8

# ## 0. 预训练模型数据读取模块

# ### 1. 处理本地数据集合 

# ####  1.1 切分数据集
# - 以9:1的比例切分数据为 训练集和验证集

# In[1]:


### 切分数据集为训练集和验证集
import random
random.seed(1)
labels = set()
fp_dev = open("data/train/train/usual_dev.tsv", "w+", encoding='utf-8')
fp = open("data/train/train/usual_train.tsv", "w+", encoding='utf-8')

file_in = open("data/train/train/usual_train.txt", "rb")
lines = file_in.read().decode("utf-8") 
lines = eval(lines)

print("total:", len(lines))
for line in lines:
    labels.add(line['label'])
    if random.random() > 0.1:
        fp.write(str(line['id']) + '\t' + line['content'] + '\t' + line['label'] + '\n')
    else:
        fp_dev.write(str(line['id']) + '\t' + line['content'] + '\t' + line['label'] + '\n')
fp.close()
labels = list(labels)
labels.sort()
print("labels:", labels)


# #### 1.2 Test
# - 将原test数据集处理为tsv文件

# In[2]:


### 修改test集
fp_test_1 = open("data/train/train/usual_test.tsv", "w+", encoding='utf-8')
fp_in = open("data/eval/eval/usual_eval.txt", "rb")
lines = fp_in.read().decode("utf-8") 
lines = eval(lines)
for line in lines:
    fp_test_1.write(str(line['id']) + '\t' + line['content'] + '\n')


# ### 2. 读取本地数据

# ### 2.1 导入相关包

# In[3]:


import torch
import logging
import os
from transformers import AutoTokenizer
import config
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


# In[4]:


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# ### 2.2 定义存储数据类型

# In[5]:


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


# ### 2.3 定义数据处理类型

# In[6]:


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        file_in = open(input_file, "rb")
        lines = []
        for line in file_in:
            lines.append(line.decode("utf-8").split("\t"))
        return lines


# In[15]:


class ClassifyProcessor(DataProcessor):
    """Processor for the Discourage data set ."""

    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "usual_train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "usual_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "usual_dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "usual_test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        tmp = list(self.labels)
        tmp.sort()
        return tmp

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1].strip()
            if len(line) < 3:
                label = None
            else:
                label = line[2].strip()
            text_b = None
            self.labels.add(label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


# ### 2.4 定义数据转换为特征函数

# In[19]:


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        if example.label is not None:
            label_id = label_map[example.label]
        else:
            label_id = None
        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [x for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if example.label is not None:
                logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id))
    return features


# ### 2.5 定义截断函数

# In[20]:


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# ### 2.6 定义生成数据类

# In[21]:


####
# 按这个顺序 all_input_ids, all_input_mask, all_segment_ids, all_label_ids 生成数据
####
class DataGenerate():
    def __init__(self):
        self.processor = ClassifyProcessor()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        self.train_examples = None
        self.num_train_steps = None
        
        self.train_data_loader = None
        self.test_data_loader = None
        self.dev_data_loader = None
        
#         self.get_train_loader()
        pass

    def get_train_loader(self):
        if self.train_data_loader is None:
            if self.train_examples is None:
                self.train_examples = self.processor.get_train_examples(config.data_dir)
            label_list = self.processor.get_labels()
            self.labels_list = label_list
            train_features = convert_examples_to_features(
                self.train_examples, self.get_labels(), config.max_seq_length, self.tokenizer)
            self.num_train_steps = int(
                len(self.train_examples) / config.train_batch_size
                / config.gradient_accumulation_steps * config.num_train_epochs)
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(self.train_examples))
            logger.info("  Batch size = %d", config.train_batch_size)
            logger.info("  Num steps = %d", self.num_train_steps)
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                
            train_sampler = RandomSampler(train_data)
            self.train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=config.train_batch_size)
        return self.train_data_loader

    def get_dev_loader(self):
        if self.dev_data_loader is None:
            dev_examples = self.processor.get_dev_examples(config.data_dir)
            dev_features = convert_examples_to_features(
                dev_examples, self.get_labels(), config.max_seq_length, self.tokenizer)
            all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)
            dev_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            self.dev_data_loader = DataLoader(dev_data, batch_size=config.dev_batch_size)
            
        return self.dev_data_loader

    def get_test_loader(self):
        if self.test_data_loader is None:
            test_examples = self.processor.get_test_examples(config.data_dir)
            test_features = convert_examples_to_features(
                test_examples, self.get_labels(), config.max_seq_length, self.tokenizer)
            all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
            # all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
            test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
            self.test_data_loader = DataLoader(test_data, batch_size=config.test_batch_size)
            
        return self.test_data_loader

    def get_labels(self):
        return self.labels_list

    def get_num_train_steps(self):
        return self.num_train_steps


# ### 2.7 实例化数据类

# In[22]:


data_generator = DataGenerate()


# ### 2.8 数据加载器

# In[24]:


####
# 按这个顺序 all_input_ids, all_input_mask, all_segment_ids, all_label_ids 生成数据
####
train_loader = data_generator.get_train_loader()
dev_loader = data_generator.get_dev_loader()
test_loader = data_generator.get_test_loader()


# ### 2.9 展示数据

# In[36]:


it = next(iter(dev_loader))
all_input_ids, all_input_mask, all_segment_ids, all_label_ids = it
print(data_generator.tokenizer.decode(all_input_ids[0], skip_special_tokens=True))
print(data_generator.get_labels()[all_label_ids[0]])


# In[35]:


help(data_generator.tokenizer.decode)


# In[ ]:




