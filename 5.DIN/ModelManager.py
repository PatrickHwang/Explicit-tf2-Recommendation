import argparse
import random

import os
import json

from Tools import EarlyStopper
from CustomLayers import *


class ModelManager:

    def __init__(self,categorical_features=['uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'i_goods_id', 'i_shop_id', 'i_cate_id', 'itag4'],
                 continuous_features=['itag4_origin','itag4_square','itag4_cube'],
                 behavior_series_features=['visited_goods_ids','visited_shop_ids','visited_cate_ids'],
                 embedding_dims=16, lr=0.001, label_name='ctr',
                 checkpoint_dir='model/checkpoint',tensorboard_dir='model/tensorboard',
                 data_dir='data/', output_dir='model/output', n_threads=-1, shuffle=50, batch=100, epochs=30,
                 restore=False, num_trials=3,layer='fm_ranking',checkpoint_interval=True,allow_continuous=False,model_params={'type':'vec'},
                 regularization_factor=0.01):
        # 模型参数和存储路径
        self.embedding_dims = embedding_dims
        self.lr = lr
        self.label_name = label_name
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_dir=tensorboard_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_threads = n_threads
        self.shuffle = shuffle
        self.batch = batch
        self.epochs = epochs
        self.restore = restore
        self.num_trials = num_trials
        self.checkpoint_interval=checkpoint_interval
        self.model_params=model_params
        self.allow_continuous=allow_continuous
        self.regularization_factor=regularization_factor

        # 初始化读取信息
        self.set_feature_names(continuous_features,categorical_features, behavior_series_features,label_name)
        self.make_layer_choice(layer_name=layer,model_params=model_params)

        # 初始化类时一同初始化建模所需对象
        self.init_model()
        self.init_loss()
        self.init_opt()
        self.init_metric()
        self.init_save_checkpoint()
        self.train_ds=self.init_dataset('train')
        self.test_ds=self.init_dataset('test')

    def load_json_info(self,json_path):
        with open(json_path,'rb') as f:
            self.feature_info=json.load(f)


    def set_feature_names(self, continuous_features=None,categorical_features=None, behavior_series_features=None,label_name=None):
        if continuous_features:
            self.continuous_features = continuous_features
        if categorical_features:
            self.categorical_features = categorical_features
        if behavior_series_features:
            self.behavior_series_features=behavior_series_features
        if label_name:
            self.label_name=label_name
        self.feature_names=self.continuous_features+self.categorical_features+self.behavior_series_features

    def make_layer_choice(self,layer_name='fm_ranking',model_params={}):
        if layer_name=='fm_ranking':
            self.layer=FMRankingLayer(feature_names=self.categorical_features,feature_dims=150000,embedding_dims=self.embedding_dims)
        if layer_name=='din_layer':
            self.layer=DINLayer(embedding_dims=3)

        else:
            raise ValueError("不在可用的模型范围内")

    # 定义model
    def init_model(self, layer=None, max_length=6):
        layer = layer if layer else self.layer
        # 输入
        input_dic = {}
        for feature in self.categorical_features:
            input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
        for feature in self.continuous_features:
            input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
        for feature in self.behavior_series_features:
            input_dic[feature] = tf.keras.Input(shape=(max_length,), name=feature, dtype=tf.int64)

        output = layer(input_dic)
        self.model = tf.keras.Model(input_dic, output)
        self.model.summary()

    # 定义loss
    def init_loss(self):
        self.loss = tf.keras.losses.BinaryCrossentropy()

    # 定义optam
    def init_opt(self):
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)

    # 定义metric
    def init_metric(self):
        self.metric_auc = tf.keras.metrics.AUC(name="auc")
        self.metric_loss = tf.keras.metrics.Mean(name="loss")

    # 保存模型
    def init_save_checkpoint(self):
        checkpoint = tf.train.Checkpoint(optimizer=self.opt, model=self.model)
        self.manager = tf.train.CheckpointManager(
            checkpoint,
            directory=self.checkpoint_dir,
            max_to_keep=3,
            checkpoint_interval=self.checkpoint_interval,
            step_counter=self.opt.iterations)

    # 准备输入数据
    def init_dataset(self, mode='train', data_dir=None, feature_names=None, label_name=None, max_length=6):
        assert mode in ('train', 'test')
        self.set_feature_names(feature_names, label_name)
        data_dir = data_dir if data_dir else self.data_dir

        def _parse_example(example):
            record = {}
            record[self.label_name] = tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
            for feature_name in self.feature_names:
                if feature_name in self.behavior_series_features:
                    record[feature_name] = tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.int64, allow_missing=True)
                elif feature_name in self.categorical_features:
                    record[feature_name] = tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
                elif feature_name in self.continuous_features:
                    record[feature_name] = tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
            feats = tf.io.parse_single_example(example, record)
            return feats

        file_name_list = [file for file in os.listdir(data_dir) if mode in file]
        files = [os.path.join(data_dir, file_name) for file_name in file_name_list]

        dataset = tf.data.Dataset.list_files(files)
        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length=self.n_threads)
        if mode == 'train':
            dataset.shuffle(self.shuffle)
        dataset = dataset.map(_parse_example,
                              num_parallel_calls=self.n_threads)
        dataset = dataset.padded_batch(self.batch,
                                       padded_shapes={f: [max_length] if f in self.behavior_series_features else [1] for
                                                      f in self.feature_names})

        dataset = dataset.prefetch(buffer_size=1)
        return dataset

    # train
    def train_step(self, ds,epoch,summary_writer):
        # start
        def train_loop_begin():
            self.metric_auc.reset_states()
            self.metric_loss.reset_states()

        # end
        def train_loop_end():
            result = {
                self.metric_auc.name: self.metric_auc.result().numpy(),
                self.metric_loss.name: self.metric_loss.result().numpy()
            }
            return result

        # loop
        def train_loop(inputs):
            with tf.GradientTape() as tape:
                target = inputs.pop(self.label_name)
                logits = self.model(inputs, training=True)

                feature_ids_list = []
                for feature in self.model.layers[-1].user_and_context_categorical_features + self.model.layers[
                    -1].item_categorical_features:
                    feature_ids_list.append(inputs[feature])

                # behavior_series_features 需要变成一维向量
                for feature in self.model.layers[-1].behavior_series_features:
                    feature_ids_list.append(tf.reshape(inputs[feature], [-1]))

                # 将所有的id合并到一个向量中并去重
                all_ids = tf.concat(feature_ids_list, axis=0)
                all_ids = tf.unique(all_ids).y

                # L2 regularization for used embeddings
                used_embeddings = tf.gather(self.model.layers[-1].embed.weights[0], all_ids)
                l2_reg = tf.nn.l2_loss(used_embeddings) * self.regularization_factor

                loss_value = tf.reduce_sum(self.loss(target, logits["output"])) + l2_reg
                gradients = tape.gradient(loss_value, self.model.trainable_variables)
                self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

                self.metric_loss.update_state(loss_value)
                self.metric_auc.update_state(target, logits["output"])

        # training
        step = 0
        inputs = iter(sorted(ds,key=lambda x:random.random()))
        train_loop_begin()
        while True:
            batch_data = next(inputs, None)
            if not batch_data:
                print(f"The train dataset iterator is exhausted after {step} steps.")
                break
            train_loop(batch_data)  # yield的是一个batch的数据
            step += 1
            if step % 500 == 0:
                result = train_loop_end()
                print(f"train over-{step} : {result}")
                with summary_writer.as_default():  # 指定记录器
                    tf.summary.scalar("train_auc", result['auc'], step=(epoch*3200+step))
                    tf.summary.scalar("train_loss", result['loss'], step=(epoch*3200+step))
        result = train_loop_end()
        print(f"train over-{step} : {result}")
        return result

    # eval
    def eval_step(self, ds):
        # start
        def eval_loop_begin():
            self.metric_auc.reset_states()
            self.metric_loss.reset_states()

        # end
        def eval_loop_end():
            result = {self.metric_auc.name: self.metric_auc.result().numpy(),
                      self.metric_loss.name:self.metric_loss.result().numpy()}
            return result

        # loop
        def eval_loop(inputs):
            target = inputs.pop(self.label_name)
            logits = self.model(inputs, training=False)
            scaled_loss = tf.reduce_sum(self.loss(target, logits["output"]))
            self.metric_loss.update_state(scaled_loss)
            self.metric_auc.update_state(target, logits["output"])

        # eval
        step = 0
        inputs = iter(ds)
        eval_loop_begin()
        while True:
            batch_data = next(inputs, None)
            if not batch_data:
                print(f"The eval dataset iterator is exhausted after {step} steps.")
                break
            eval_loop(batch_data)
            step += 1
            if step % 500 == 0:
                result = eval_loop_end()
                print(f"eval over-{step} : {result}")
        result = eval_loop_end()
        print(f"eval over-{step} : {result}")
        return result

    # run
    def run(self, train_ds=None, test_ds=None, mode="train_and_eval", incremental=False):
        train_ds=train_ds if train_ds else self.train_ds
        test_ds=test_ds if test_ds else self.test_ds
        if incremental:
            self.load_model()
            self.model = self.imported
        if self.restore:
            self.manager.restore_or_initialize()
        early_stopper = EarlyStopper(num_trials=self.num_trials)
        if mode == "train_and_eval":
            # 清理tensorboard旧文件并初始化writer
            for file in os.listdir(self.tensorboard_dir):
                os.remove(os.path.join(self.tensorboard_dir, file))
            summary_writer = tf.summary.create_file_writer(self.tensorboard_dir)  # 实例化记录器
            tf.summary.trace_on(profiler=True)  # 开启Trace（可选）
            for epoch in range(self.epochs):
                print(f"=====epoch: {epoch}=====")
                train_result = self.train_step(train_ds,epoch,summary_writer)
                eval_result = self.eval_step(test_ds)
                with summary_writer.as_default():  # 指定记录器
                    tf.summary.scalar("eval_auc", eval_result['auc'], step=epoch)
                    tf.summary.scalar("eval_loss", eval_result['loss'], step=epoch)

                if not early_stopper.is_continuable(eval_result[self.metric_auc.name]):
                    print(
                        f'validation best accuracy {early_stopper.best_metric} at epoch {epoch - early_stopper.num_trials}')
                    break

                if eval_result[self.metric_auc.name] == early_stopper.best_metric:
                    self.export_model()
                    self.manager.save(checkpoint_number=epoch,
                                      check_interval=True)
        if mode == "train":
            for epoch in range(self.epochs):
                print(f"=====epoch: {epoch}=====")
                train_result = self.train_step(train_ds)
                if train_result[self.metric_auc.name] > 0.5:
                    self.manager.save(checkpoint_number=epoch,
                                      check_interval=True)
        if mode == "eval":
            eval_result = self.eval_step(test_ds)

    # export
    def export_model(self):
        tf.saved_model.save(self.model, self.output_dir)

    def load_model(self):
        self.imported = tf.saved_model.load(self.output_dir)
        # self.f = imported.signatures["serving_default"]

    # infer
    def infer(self, x):
        result = self.imported(x)
        return result

    def load_checkpoint_and_eval(self,eval=True,checkpoint_path=None):
        if not checkpoint_path:
            checkpoint_path=self.checkpoint_dir
        self.manager.restore_or_initialize()
        checkpoint = tf.train.Checkpoint(myAwesomeModel=self.model)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
        if eval:
            self.run(mode="eval")
        self.imported=self.model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse args.')
    parser.add_argument('--feature_names', type=str, default="['user_tag1', 'user_tag2', 'item_tag1', 'item_tag2', 'item_tag3']")
    parser.add_argument('--json_path', type=str, default='data/generated/data_info.json')
    parser.add_argument('--embedding_dims', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--label_name', type=str, default='label')
    parser.add_argument('--checkpoint_dir', type=str, default='retrieval_model/checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/generated-non-time-sequence')
    parser.add_argument('--output_dir', type=str, default='retrieval_model/output')
    parser.add_argument('--tensorboad_dir', type=str, default='retrieval_model/tensorboard')
    parser.add_argument('--n_threads', type=int, default=-1)
    parser.add_argument('--shuffle', type=int, default=5)
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--restore', type=bool, default=False)
    parser.add_argument('--num_trials', type=int, default=3)
    parser.add_argument('--layer', type=str, default='dssm_double_tower')
    parser.add_argument('--checkpoint_interval', type=bool, default=True)
    parser.add_argument('--model_params',type=str,default='{}')
    args = parser.parse_args()
    model_params=eval(args.model_params)
    assert type(model_params)==dict,"模型参数输入错误，须为字典"
    """
    mm = ModelManager(categorical_features=['item_tag1', 'item_tag2', 'item_tag3'])
    input = {'item_tag1': tf.constant([0, 1, 2]), 'item_tag2': tf.constant([3, 4, 5]),
             'item_tag3': tf.constant([6, 7, 8])}
    print(mm.model(input))
    """
    mm = ModelManager(layer='din_layer')
    input = {
        'uid': tf.constant([0, 1, 2]),
        'utag1': tf.constant([6, 7, 8]),
        'utag2': tf.constant([9, 10, 11]),
        'utag3': tf.constant([12, 13, 14]),
        'utag4': tf.constant([15, 16, 17]),
        'i_goods_id': tf.constant([18, 19, 20]),
        'i_shop_id': tf.constant([21, 22, 23]),
        'i_cate_id': tf.constant([24, 25, 26]),
        'itag4_origin': tf.constant([0.2, 7.8, 4.9]),
        'itag4_square': tf.constant([5.3, 1.2, 8.0]),
        'itag4_cube': tf.constant([-3.8, -19.6, 4.2]),
        'visited_goods_ids': tf.constant([[30, 31, 32, 0, 0, 0], [33, 34, 35, 36, 0, 0], [37, 38, 39, 40, 41, 0]]),
        'visited_shop_ids': tf.constant([[42, 43, 44, 0, 0, 0], [45, 46, 47, 48, 0, 0], [49, 50, 51, 52, 53, 0]]),
        'visited_cate_ids': tf.constant([[54, 55, 56, 0, 0, 0], [57, 58, 59, 60, 0, 0], [61, 62, 63, 64, 65, 0]])
    }
    print(mm.model(input))

