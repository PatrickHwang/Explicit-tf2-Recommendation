from ModelBase import ModelBase
from MLPLayer import MLPLayer
import tensorflow as tf
import numpy as np
import argparse
import os



class MatrixCF(ModelBase):
    def __init__(self, feature_dims=9000, embedding_size=16, lr=0.01, label_name='ctr',
                 checkpoint_dir='model/checkpoint',
                 data_dir='data/generated', output_dir='model/output', n_threads=-1, shuffle=50, batch=16, epochs=30,
                 restore=False, num_trials=3):
        super().__init__(feature_dims, embedding_size, lr, label_name,checkpoint_dir,
                 data_dir, output_dir, n_threads, shuffle, batch, epochs,restore, num_trials)

    def init_model(self):
        user = tf.keras.Input(shape=(1,), name='user', dtype=tf.int64)
        item = tf.keras.Input(shape=(1,), name='item', dtype=tf.int64)
        embed = tf.keras.layers.Embedding(self.feature_dims,
                                          self.embedding_size,
                                          embeddings_regularizer="l2")
        user_emb = embed(user)
        item_emb = embed(item)
        # 使用余弦相似度,并把值域从[-1,1]线性映射成[0,1]
        logit = tf.keras.losses.cosine_similarity(user_emb, item_emb)
        logit = (1 + logit) / 2
        # 或者直接取余弦绝对值,值域[0,1]
        # logit = abs(tf.keras.losses.cosine_similarity(user_emb,item_emb))
        self.model = tf.keras.Model({
            "user": user,
            "item": item
        }, {
            "pred": logit,
            "user_emb": tf.identity(user_emb, name="user_emb"),
            "item_emb": tf.identity(item_emb, name="item_emb")
        })
        self.model.summary()

    def init_dataset(self, type, file_dir=None, shuffle=True):
        assert type in ['train', 'test'], 'wrong type!Should be either train or test'
        if not file_dir:
            file_dir = self.data_dir

        def _parse_example(example):
            feats = {}
            feats["ctr"] = tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
            feats["user"] = tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
            feats["item"] = tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
            feats = tf.io.parse_single_example(example, feats)
            return feats

        try:
            files = [os.path.join(file_dir, file_name) for file_name in os.listdir(file_dir) if
                     ((not file_name.endswith('.json')) and type in file_name)]
        except NotADirectoryError:
            print('the data path is not a directory but a file')
            files = [file_dir]
        dataset = tf.data.Dataset.list_files(files)
        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length=self.n_threads)
        if shuffle:
            dataset.shuffle(self.shuffle)
        dataset = dataset.map(_parse_example,
                              num_parallel_calls=self.n_threads)
        dataset = dataset.batch(self.batch)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset


class NeuralCollaborativeFiltering(ModelBase):
    def __init__(self, feature_dims=9000, embedding_size=16, lr=0.01, label_name='ctr',
                 checkpoint_dir='model/checkpoint',
                 data_dir='data/generated', output_dir='model/output', n_threads=-1, shuffle=50, batch=16, epochs=30,
                 restore=False, num_trials=3):
        super().__init__(feature_dims, embedding_size, lr, label_name,checkpoint_dir,
                 data_dir, output_dir, n_threads, shuffle, batch, epochs,restore, num_trials)

    def init_model(self):
        user = tf.keras.Input(shape=(1,), name='user', dtype=tf.int64)
        item = tf.keras.Input(shape=(1,), name='item', dtype=tf.int64)
        mf_embed = tf.keras.layers.Embedding(self.feature_dims,
                                          self.embedding_size,
                                          embeddings_regularizer="l2")
        mlp_embed= tf.keras.layers.Embedding(self.feature_dims,
                                          self.embedding_size,
                                          embeddings_regularizer="l2")
        mf_user_emb = tf.keras.layers.Flatten()(mf_embed(user))
        mf_item_emb = tf.keras.layers.Flatten()(mf_embed(item))
        mlp_user_emb = tf.keras.layers.Flatten()(mlp_embed(user))
        mlp_item_emb = tf.keras.layers.Flatten()(mlp_embed(item))
        
        mf_vector=tf.multiply(mf_user_emb,mf_item_emb)

        mlp_vector=tf.concat([mlp_user_emb,mlp_item_emb],axis=1)
        mlp_vector=MLPLayer(units=[10], activation='relu', is_batch_norm=True)(mlp_vector)


        merge_vector=tf.concat([mf_vector,mlp_vector],axis=1)
        merge_vector = MLPLayer(units=[8], activation='relu')(merge_vector)
        logit = MLPLayer(units=[1], activation='sigmoid')(merge_vector)

        user_embedding=tf.concat([mf_user_emb,mlp_user_emb],axis=1)
        item_embedding = tf.concat([mf_item_emb, mlp_item_emb], axis=1)


        self.model = tf.keras.Model({
            "user": user,
            "item": item
        }, {
            "pred": logit,
            "user_emb": tf.identity(user_embedding, name="user_emb"),
            "item_emb": tf.identity(item_embedding, name="item_emb")
        })
        self.model.summary()

    def init_dataset(self, type, file_dir=None, shuffle=True):
        assert type in ['train', 'test'], 'wrong type!Should be either train or test'
        if not file_dir:
            file_dir = self.data_dir

        def _parse_example(example):
            feats = {}
            feats["ctr"] = tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
            feats["user"] = tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
            feats["item"] = tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
            feats = tf.io.parse_single_example(example, feats)
            return feats

        try:
            files = [os.path.join(file_dir, file_name) for file_name in os.listdir(file_dir) if
                     ((not file_name.endswith('.json')) and type in file_name)]
        except NotADirectoryError:
            print('the data path is not a directory but a file')
            files = [file_dir]
        dataset = tf.data.Dataset.list_files(files)
        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length=self.n_threads)
        if shuffle:
            dataset.shuffle(self.shuffle)
        dataset = dataset.map(_parse_example,
                              num_parallel_calls=self.n_threads)
        dataset = dataset.batch(self.batch)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse args.')
    parser.add_argument('--feature_dims', type=int, default=9000)
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--label_name', type=str, default='ctr')
    parser.add_argument('--checkpoint_dir', type=str, default='model/checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/generated')
    parser.add_argument('--output_dir', type=str, default='model/output')
    parser.add_argument('--n_threads', type=int, default=-1)
    parser.add_argument('--shuffle', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--restore', type=bool, default=False)
    parser.add_argument('--num_trials', type=int, default=3)
    args = parser.parse_args()
    ncf = NeuralCollaborativeFiltering(feature_dims=args.feature_dims, embedding_size=args.embedding_size, lr=args.lr,
                   label_name=args.label_name, checkpoint_dir=args.checkpoint_dir,
                   data_dir=args.data_dir, output_dir=args.output_dir, n_threads=args.n_threads, shuffle=args.shuffle,
                   batch=args.batch, epochs=args.epochs, restore=args.restore, num_trials=args.num_trials)
    train_ds = ncf.init_dataset('train')
    test_ds = ncf.init_dataset('test', shuffle=False)
    # train
    ncf.run(train_ds, test_ds, mode="train_and_eval", incremental=False)
    # save
    ncf.export_model()