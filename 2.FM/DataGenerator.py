from absl import flags
from absl import app
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from Tools import CustomTFWriter

assert tf.__version__[0] == '2', 'model is built by tensorflow2,please check tf version first.'

FLAGS = flags.FLAGS
flags.DEFINE_string('item_path', default='data/raw/item_feature.dat', help='item_path')
flags.DEFINE_string('user_path', default='data/raw/user_feature.dat', help='user_path')
flags.DEFINE_string('main_path', default='data/raw/shop.dat', help='main_path')
flags.DEFINE_string('output_path', default='data/generated', help='output_path')
flags.DEFINE_string('encode_columns', default="['user_tag1', 'user_tag2', 'item_tag1', 'item_tag2', 'item_tag3']",
                    help='output_path')
flags.DEFINE_string('data_name', default="e-Commerce", help='output_path')

# 只拼接完整记录，只用特征建模，不使用uid和item_id做embedding并建模
"""
不使用user_id和item_id进行建模的几点理由：
1.全量user_id和item_id是千万级别，embedding需要存储的参数量太大
2.非热门用户和商品在样本中出现的频率过于稀疏(大多数从未出现)，embedding不会得到更新或无法得到充分更新。而长尾是推荐场景的常态。
3.那么，一个从未出现的user或item的embedding最终值就和初始化时一致，如果不希望初始化的差异造成预测结果差异，就只能使用相同的初始化值。
然而，用户的热门和冷门并不是永远不变的，难以取得合适的划分原则。维护一个动态的“热门列表”和“冷门列表”并更新到embedding并不简单
4.商品更新/下架，用户注册/注销
"""


class DataGenerator:
    def __init__(self, output_path='data/generated', data_name='e-Commerce',
                 item_path='data/raw/item_feature.dat',
                 user_path='data/raw/user_feature.dat',
                 main_path='data/raw/shop.dat',
                 encode_columns=['user_tag1', 'user_tag2', 'item_tag1', 'item_tag2', 'item_tag3'],
                 user_feature_num=2):
        self.output_path = output_path
        self.item_path = item_path
        self.user_path = user_path
        self.main_path = main_path
        self.recorder = {}
        self.feature_dims = None
        self.feature_offsets = None
        self.df_merged = None
        self.feature_values_cnt = None
        self.encode_columns = encode_columns
        self.data_name = data_name
        self.user_features = encode_columns[:user_feature_num]
        self.item_features = encode_columns[user_feature_num:]

    def read_and_merge(self, item_path=None, user_path=None, main_path=None):
        # pd.merge拼接样本表和画像信息，得到self.df_merged
        if not item_path:
            item_path = self.item_path
        if not user_path:
            user_path = self.user_path
        if not main_path:
            main_path = self.main_path
        self.df_item = pd.read_csv(item_path, names=['item_id', 'item_tag1', 'item_tag2', 'item_tag3'])
        self.df_user = pd.read_csv(user_path, names=['user_id', 'user_tag1', 'user_tag2'])
        df_main = pd.read_csv(main_path, names=['timestamp', 'user_id', 'item_id', 'label'])
        df_merged = pd.merge(df_main, self.df_user, on=['user_id'], how='left')
        df_merged = pd.merge(df_merged, self.df_item, on=['item_id'], how='left')
        print(df_merged.notnull().sum())
        # notebook中分析下各特征的空缺率和取值占比，对有空缺值的记录要么填补要么丢弃
        self.df_merged = df_merged[df_merged.notnull().sum(axis=1) == df_merged.shape[1]]
        split_threshold = np.percentile(df_merged['timestamp'], [80])[0]
        self.df_merged['data_type'] = 'train'
        self.df_merged.loc[df_merged['timestamp'] >= split_threshold, 'data_type'] = 'test'
        # self.df_merged['data_type']=np.random.choice(['train','test'],self.df_merged.shape[0],p=[0.8,0.2])
        print(self.df_merged['data_type'].value_counts() / self.df_merged['data_type'].count())

    def get_feature_dims(self):
        # 记录各画像特征的取值类别数，并计算各特征应取的偏移量
        self.feature_dims = [len(self.df_merged[col].unique()) for col in self.encode_columns]
        self.feature_offsets = (0, *np.cumsum(self.feature_dims[:-1]))
        self.feature_values_cnt = sum(self.feature_dims)

    def encode_and_record(self):
        # 独热处理，记得加上特征偏移量
        le = LabelEncoder()
        for index, col in enumerate(self.encode_columns):
            offset = self.feature_offsets[index]
            self.df_merged[col] = le.fit_transform(self.df_merged[col]) + offset
            self.recorder[col] = {str(value): int(i + offset) for i, value in enumerate(le.classes_)}
        # print(self.recorder)

    def translate_profiles(self):
        # 为了线上服务通过user_id和item_id取到embedding的输入码
        self.user_profile = {}
        self.item_profile = {}
        for record in self.df_user.values:
            user_id, *user_values = record
            self.user_profile[str(user_id)] = [self.recorder[name][str(value)] for name, value in
                                          zip(self.user_features, user_values)]
        for record in self.df_item.values:
            item_id, *item_values = record
            self.item_profile[str(item_id)] = [self.recorder[name][str(value)] for name, value in
                                          zip(self.item_features, item_values)]

    def write_tf_records(self):
        # 写入tf数据文件
        counter = {'train': 0, 'test': 0}
        custom_writer = CustomTFWriter(data_name=self.data_name, tf_doc_record_limit=200000,
                                       output_path=self.output_path)
        all_cols = self.encode_columns + ['data_type', 'label']
        for line in self.df_merged[all_cols].values:
            data_type, label = line[-2:]
            features = line[:-2]
            sample = {}
            sample["label"] = tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
            for i, feature in enumerate(self.encode_columns):
                sample[feature] = tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[features[i]]))
            sample = tf.train.Example(features=tf.train.Features(feature=sample))
            if data_type == 'train':
                custom_writer.write(sample, 'train')
            elif data_type == 'test':
                custom_writer.write(sample, 'test')
            counter[data_type] += 1
        print(counter)

    def dump_dicts(self):
        with open(self.output_path + '/feature_dict.json', 'w') as f:
            json.dump(self.recorder, f)
        with open(self.output_path + '/data_info.json', 'w') as f:
            json.dump([list(map(int,self.feature_dims)), list(map(int,self.feature_offsets)), int(self.feature_values_cnt)], f)
        with open(self.output_path + '/user_profile.json', 'w') as f:
            json.dump(self.user_profile, f)
        with open(self.output_path + '/item_profile.json', 'w') as f:
            json.dump(self.item_profile, f)

    def clean_output_folder(self):
        # 运行前先清空旧文件
        for file in os.listdir(self.output_path):
            os.remove(os.path.join(self.output_path, file))

    def run_all(self):
        self.clean_output_folder()
        self.read_and_merge()
        self.get_feature_dims()
        self.encode_and_record()
        self.translate_profiles()
        self.write_tf_records()
        self.dump_dicts()
        print('done!')


def run(_):
    dg = DataGenerator(output_path=FLAGS.output_path,
                       item_path=FLAGS.item_path,
                       user_path=FLAGS.user_path,
                       main_path=FLAGS.main_path,
                       encode_columns=eval(FLAGS.encode_columns),
                       data_name=FLAGS.data_name)
    dg.run_all()


if __name__ == '__main__':
    app.run(run)
