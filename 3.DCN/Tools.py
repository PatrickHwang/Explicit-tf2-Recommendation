import numpy as np

import tensorflow as tf

assert tf.__version__[0] == '2', 'model is built by tensorflow2,please check tf version first.'


class CustomTFWriter:
    def __init__(self, data_name, tf_doc_record_limit,output_path='data/generated-non-time-sequence'):
        self.data_name = data_name
        self.train_writer = None
        self.test_writer = None
        self.output_path=output_path
        self.train_doc_index = 1
        self.test_doc_index = 1
        self.tf_doc_record_limit = tf_doc_record_limit
        self.train_inner_index = 1
        self.test_inner_index = 1
        for type in ['train', 'test']:
            # 初始化一下训练和测试的写入器
            self._update_tf_writer(type)

    def _update_tf_writer(self, type):
        # 用于更新writer，关闭上一个writer并生成一个自增编号的新writer
        assert type in ['train', 'test'], "type could only be train or test"
        if type == 'train':
            if self.train_writer:
                self.train_writer.close()
                self.train_doc_index += 1
            writer_name = self.data_name + '-' + type + '-' + str(self.train_doc_index)
            self.train_writer = tf.io.TFRecordWriter(self.output_path+'/'+writer_name)
        if type == 'test':
            if self.test_writer:
                self.test_writer.close()
                self.test_doc_index += 1
            writer_name = self.data_name + '-' + type + '-' + str(self.test_doc_index)
            self.test_writer = tf.io.TFRecordWriter(self.output_path+'/'+writer_name)
        return None

    def write(self, sample,type):
        # 文档内部计数，满则更新writer并重置计数
        assert type in ['train', 'test'], "type could only be train or test"
        if type=='train':
            self.train_inner_index += 1
            self.train_writer.write(sample.SerializeToString())
            if self.train_inner_index >= self.tf_doc_record_limit:
                self._update_tf_writer('train')
                self.train_inner_index = 1
        else:
            self.test_inner_index += 1
            self.test_writer.write(sample.SerializeToString())
            if self.test_inner_index >= self.tf_doc_record_limit:
                self._update_tf_writer('test')
                self.test_inner_index = 1

class EarlyStopper:
    """
    跟踪模型训练中每个epoch的指标，触发早停条件时返回信号停止训练
    """
    def __init__(self, num_trials, is_higher_better=True,criterion='best'):
        assert criterion in ('best','better')
        self.criterion=criterion
        self.num_trials = num_trials
        self.trial_counter = 0
        self.is_higher_better = is_higher_better
        if is_higher_better:
            self.best_metric = 0
            self.last_metric = 0
        else:
            self.best_metric = 1e6
            self.last_metric = 1e6
        self.recorder=[]

    def is_continuable(self, metric):
        self.recorder.append(metric)
        if self.criterion=='best' and ((self.is_higher_better and metric > self.best_metric) or \
                (not self.is_higher_better and metric < self.best_metric)):
            self.best_metric = metric
            self.trial_counter = 0
            self.last_metric=metric
            return True
        elif self.criterion=='better' and ((self.is_higher_better and metric > self.last_metric) or \
                (not self.is_higher_better and metric < self.last_metric)):
            self.last_metric=metric
            self.trial_counter = 0
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            self.last_metric=metric
            return True
        else:
            self.last_metric=metric
            return False

    def check_record(self):
        if self.is_higher_better:
            index=np.argmax(self.recorder)
        else:
            index=np.argmin(self.recorder)
        return index,self.recorder