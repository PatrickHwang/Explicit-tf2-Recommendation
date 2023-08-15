import argparse
import random
import json
import os
import shutil
import tensorflow as tf

assert tf.__version__[0] == '2', 'model is built by tensorflow2,please check tf version first.'


class DataGenerator:
    def __init__(self, raw_data_path, output_path, custom_tf_writer,
                 features=[(1, 'user', str), (2, 'item', str), (3, 'ctr', float)]):
        """
        :param raw_data_path: 原始数据存放路径
        :param output_path:   目标生成数据存放路径
        :param custom_tf_writer: 包装后的tf.io.TFRecordWriter调用器
        :param features:      需要建模的字段信息，[(字段在源文件中分列后的索引，字段名称，字段类型)]
        :param i2o: 原始id -> onehot类别号
        :param o2i：onehot类别号 -> 原始id
        """
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.features = features
        self.custom_tf_writer = custom_tf_writer
        self.i2o_dict = {}

    def _parse_data(self):
        with open(self.raw_data_path, encoding="utf8") as f:
            while True:
                line = f.readline()
                if not line or len(line) == 0:
                    break
                line = line.strip().split('\t')
                sample = {}
                for index, feature_name, data_type in self.features:
                    # 解析字段信息，并根据字段类型、名称等用tf.train的子类分别包装
                    if data_type == float:
                        sample[feature_name] = tf.train.Feature(
                            float_list=tf.train.FloatList(value=[float(line[index])]))
                    elif data_type == int:
                        sample[feature_name] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(line[index])]))
                    elif data_type == str:
                        id = line[index]
                        if id not in self.i2o_dict:
                            self.i2o_dict[id] = len(self.i2o_dict)
                        sample[feature_name] = tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[self.i2o_dict[id]]))
                sample = tf.train.Example(features=tf.train.Features(feature=sample))
                # 使用tf.io.TFRecordWriter调用器，可以根据test_size划分数据，并根据单个文件的记录上限分文件写入
                self.custom_tf_writer.write(sample)

    def _dump_dicts(self):
        # 用json保存类别变量和索引的关系
        self.o2i_dict = {value: key for key, value in self.i2o_dict.items()}
        # print(len(self.o2i_dict))
        # print(len(self.i2o_dict))
        with open(self.output_path + '/i2o.json', 'w') as f:
            json.dump(self.i2o_dict, f)
        with open(self.output_path + '/o2i.json', 'w') as f:
            json.dump(self.o2i_dict, f)

    def _move_tf_records(self):
        # tf_writer不能指定写入文件的目标路径，故单独移动
        tf_records = [file for file in os.listdir() if file.startswith(self.custom_tf_writer.data_name)]
        for file in tf_records:
            shutil.move(file, self.output_path)

    def clean_output_folder(self):
        # 运行前先清空旧文件
        for file in os.listdir(self.output_path):
            os.remove(os.path.join(self.output_path, file))

    def run(self):
        # 工作流：清空文件、解析源文件数据到tf_record文件、保存映射字典为json、移动目标文件
        self.clean_output_folder()
        self._parse_data()
        self._dump_dicts()
        self.custom_tf_writer.train_writer.close()
        self.custom_tf_writer.test_writer.close()
        self._move_tf_records()
        print('data generation finished!')


class CustomTFWriter:
    def __init__(self, data_name, tf_doc_record_limit=10000, test_size=0.2):
        self.data_name = data_name
        self.train_writer = None
        self.test_writer = None
        self.train_doc_index = 1
        self.test_doc_index = 1
        self.tf_doc_record_limit = tf_doc_record_limit
        self.test_size = test_size
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
            self.train_writer = tf.io.TFRecordWriter(writer_name)
        if type == 'test':
            if self.test_writer:
                self.test_writer.close()
                self.test_doc_index += 1
            writer_name = self.data_name + '-' + type + '-' + str(self.test_doc_index)
            self.test_writer = tf.io.TFRecordWriter(writer_name)
        return None

    def write(self, sample):
        # 按设定比例随机写入训练/测试集
        # 文档内部计数，满则更新writer并重置计数
        if random.uniform(0, 1) > self.test_size:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse args.')
    parser.add_argument('--raw_data_path', type=str,
                        default="/Volumes/Beta/deepshare/CustomCoding/1.MatrixCF/data/raw/mini_news.dat")
    parser.add_argument('--output_path', type=str,
                        default="/Volumes/Beta/deepshare/CustomCoding/1.MatrixCF/data/generated")
    parser.add_argument('--features', type=str, default="[(1, 'user', str), (2, 'item', str), (3, 'ctr', float)]")
    parser.add_argument('--data_name', type=str, default='mini_news')
    parser.add_argument('--tf_doc_record_limit', type=int, default=10000)
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()
    custom_tf_writer = CustomTFWriter(args.data_name, args.tf_doc_record_limit, args.test_size)
    data_generator = DataGenerator(args.raw_data_path, args.output_path, custom_tf_writer, eval(args.features))
    data_generator.run()
