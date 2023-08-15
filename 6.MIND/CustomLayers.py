import tensorflow as tf
from tensorflow.python.ops.math_ops import MatMul
from tensorflow.python.ops.nn_ops import BiasAdd
from tensorflow.python.keras import activations, initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization

from tensorflow.python.framework.errors_impl import InvalidArgumentError as wrapError

import numpy as np

import sys

import importlib.util

spec = importlib.util.spec_from_file_location("DIN_CustomLayers", "../5.DIN/CustomLayers.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DIENLayer = module.DIENLayer

assert tf.__version__[0] == '2', 'model is built by tensorflow2,please check tf version first.'


class Dice(tf.keras.layers.Layer):
    def __init__(self, axis=-1, epsilon=1e-9):
        super(Dice, self).__init__()
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(name='alpha', shape=(input_shape[-1],), initializer='zeros')

    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)
        return self.alpha * (1.0 - x_p) * x + x_p * x


def make_mlp_layer(units, activation='PReLU', normalization='layernorm', softmax_units=-1, sigmoid_units=False):
    mlp = tf.keras.models.Sequential()
    for i, unit in enumerate(units):
        mlp.add(tf.keras.layers.Dense(unit))
        if normalization == 'batchnorm':
            mlp.add(tf.keras.layers.BatchNormalization())
        elif normalization == 'layernorm':
            mlp.add(tf.keras.layers.LayerNormalization())
        if activation == 'PReLU':
            mlp.add(tf.keras.layers.PReLU())
        elif activation == 'Dice':
            mlp.add(Dice())
        else:
            mlp.add(tf.keras.layers.Activation(activation))
    if softmax_units > 0:
        mlp.add(tf.keras.layers.Dense(softmax_units, activation='softmax'))
    elif sigmoid_units:
        mlp.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return mlp


class MultiInterestExtractorLayer(tf.keras.layers.Layer):
    """
    layer=MultiInterestExtractorLayer(4)
    batch_size,sequence_length,embedding_dims=3,5,8
    behaviors=tf.random.uniform(shape=[batch_size,sequence_length,embedding_dims], minval=0, maxval=1, dtype=tf.float32)
    sequence_mask = tf.random.uniform(shape=[batch_size,sequence_length], minval=0, maxval=2, dtype=tf.int32)
    sequence_mask = tf.cast(sequence_mask, dtype=tf.bool)
    print(layer((behaviors, sequence_mask)))
    """

    def __init__(self, capsules_num, capsules_dim, routing_rounds=3):
        super(MultiInterestExtractorLayer, self).__init__()
        self.capsules_num = capsules_num
        self.capsules_dim = capsules_dim
        self.routing_rounds = routing_rounds

    def build(self, input_shape):
        super(MultiInterestExtractorLayer, self).build(input_shape)
        embedding_dims = input_shape[0][-1]

        # 原论文 self.capsules_num=max(1, min(一个事先指定的值, int(np.log2(有效行为数))))
        # 每个用户输出的胶囊数可能不同，这意味着输出的capsules也需要有mask

        self.S = self.add_weight(shape=(embedding_dims, self.capsules_dim),
                                 initializer='random_normal',
                                 trainable=True)
        # S是连接输入的矩阵，把输入维度转换成胶囊维度
        # 在论文中，每个胶囊的S是共享的
        # caps = tf.einsum('bkl,bld->bkd', W, behaviors)  # caps: (batch_size, capsules_num, capsule_dims)
        # 这一步实现了capsules_num的共享，否则应该是bkl,bkld->bkd
        self.B = self.add_weight(shape=(self.capsules_num, input_shape[0][1]),
                                 initializer='random_normal',
                                 trainable=False)
        # B是要经过softmax加权sequence_length的参数矩阵，而且它是胶囊间独立的
        # caps = tf.einsum('bkl,bld->bkd', W, behaviors)  # caps: (batch_size, capsules_num, capsule_dims)
        # W是和B同形状的，它把hehavior按sequence_length维度加权了,结果就是squash前的caps

        # B += tf.einsum('bkd,bld->bkl', caps, behaviors)  # B: (batch_size, capsules_num, sequence_length)
        # behaviors: (batch_size, sequence_length, capsule_dims)
        # 上上式把每一个输出胶囊和每一个输入步的capsule_dims算点积，所以每一个胶囊每一个输入步都有一个相似度

        # 以上，形成循环，一切从B和输入开始，到此B获得了更新，输入不变

    def squash(self, caps):
        n = tf.norm(caps, axis=2, keepdims=True)
        n_square = tf.square(n)

        return (n_square / ((1 + n_square) * n)) * caps

    def call(self, inputs):
        behaviors, sequence_mask = inputs  # behaviors: (batch_size, sequence_length, embedding_dims)

        # Tile B for each sample in the batch
        B = tf.tile(self.B[None, :, :],
                    [tf.shape(behaviors)[0], 1, 1])  # B: (batch_size, capsules_num, sequence_length)

        # Mask padding indices' routing logit
        mask = tf.tile(sequence_mask[:, None, :],
                       [1, self.capsules_num, 1])  # mask: (batch_size, capsules_num, sequence_length)
        drop = tf.ones_like(mask, dtype=tf.float32) * tf.float32.min

        # Perform routing rounds
        behaviors = tf.matmul(behaviors, self.S)  # behaviors: (batch_size, sequence_length, capsule_dims)

        for i in range(self.routing_rounds):
            B_masked = tf.where(mask, B, drop)  # B_masked: (batch_size, capsules_num, sequence_length)
            W = tf.nn.softmax(B_masked, axis=2)  # W: (batch_size, capsules_num, sequence_length)

            if i < self.routing_rounds - 1:
                caps = tf.einsum('bkl,bld->bkd', W, behaviors)  # caps: (batch_size, capsules_num, capsule_dims)
                caps = self.squash(caps)  # caps: (batch_size, capsules_num, capsule_dims)
                B += tf.einsum('bkd,bld->bkl', caps, behaviors)  # B: (batch_size, capsules_num, sequence_length)
            else:
                caps = tf.einsum('bkl,bld->bkd', W, behaviors)  # caps: (batch_size, capsules_num, capsule_dims)
                caps = self.squash(caps)  # caps: (batch_size, capsules_num, capsule_dims)

        return caps  # caps: (batch_size, capsules_num, capsule_dims)


class LabelAwareAttention(tf.keras.layers.Layer):
    def call(self, mlped_capsules, capsules_mask, X_item):
        # 计算每个 capsule 与每个 item 的内积，得到一个分数矩阵
        scores = tf.einsum('bck,bsk->bcs', mlped_capsules, X_item)  # shape: (batch_size, capsules_num, num_samples)

        # 使用 mask 屏蔽无效的 capsules
        mask = ~capsules_mask[:, :, None]  # Reverse the mask
        drop = tf.ones_like(mask, dtype=tf.float32) * tf.float32.min  # Use a very small value to replace 0
        scores_masked = tf.where(mask, scores, drop)  # Mask the scores

        # 对每个 item，对其分数进行 softmax 操作，得到每个 capsule 的权重
        weights = tf.nn.softmax(scores_masked, axis=1)  # shape: (batch_size, capsules_num, num_samples)

        # 用每个 capsule 的权重对 mlped_capsules 进行加权求和
        attention_output = tf.einsum('bck,bcs->bsk', mlped_capsules,
                                     weights)  # shape: (batch_size, num_samples, capsules_dim)

        return attention_output


class MINDLayer(tf.keras.layers.Layer):
    def __init__(self, user_and_context_categorical_features=['uid', 'utag1', 'utag2', 'utag3', 'utag4'],
                 item_categorical_features=['label_goods_ids', 'label_shop_ids', 'label_cate_ids'],
                 behavior_series_features=['visited_goods_ids', 'visited_shop_ids', 'visited_cate_ids'],
                 feature_dims=160000, embedding_dims=16,
                 activation='ReLU', padding_index=0, capsules_num=4):
        super(MINDLayer, self).__init__()
        self.user_and_context_categorical_features = user_and_context_categorical_features
        self.item_categorical_features = item_categorical_features
        assert len(item_categorical_features) == len(
            behavior_series_features), "Features to be interacted should match in item and behavior series"
        self.behavior_series_features = behavior_series_features
        self.feature_dims = feature_dims
        self.embedding_dims = embedding_dims

        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims)

        capsules_dim = embedding_dims * len(item_categorical_features)

        # 和兴趣胶囊拼接的用户特征，限制一下维度
        self.context_mlp = tf.keras.layers.Dense(capsules_dim // 2, activation='sigmoid')

        self.context_bn = tf.keras.layers.BatchNormalization()

        self.final_mlp = make_mlp_layer([capsules_dim * 4, capsules_dim], activation=activation)

        self.padding_index = padding_index

        self.capsules_num = capsules_num

        self.MIE_layer = MultiInterestExtractorLayer(capsules_num=capsules_num, capsules_dim=capsules_dim)

        self.LAA_layer = LabelAwareAttention()

    def build(self, input_shape):
        super().build(input_shape)

    def generate_interest_capsules(self, inputs):
        # 确定batch中每个样本行为序列的mask
        sequence_mask = inputs[self.behavior_series_features[0]] == self.padding_index

        # 原论文是做平均池化，这里改为更常见的concat拼接
        X_series = tf.stack([inputs[feature] for feature in self.behavior_series_features],
                            axis=2)  # (batch_size, seq_length, feature_cols)
        batch_size, seq_length, feature_cols = X_series.shape
        X_series = tf.reshape(X_series, [-1, feature_cols * seq_length])  # (batch_size,feature_cols*seq_length)
        X_series = self.embed(X_series)  # (batch_size,feature_cols*seq_length,emb_dim)
        X_series = tf.reshape(X_series, [-1, seq_length,
                                         feature_cols * self.embedding_dims])  # (batch_size,seq_length,feature_cols*emb_dim)

        interest_capsules = self.MIE_layer((X_series, ~sequence_mask))

        valid_behavior_num = tf.reduce_sum(tf.cast(~sequence_mask, tf.float32), axis=1)  # batch_size
        valid_capsules_num = tf.math.log(valid_behavior_num) / tf.math.log(2.)  # batch_size
        valid_capsules_num = tf.minimum(valid_capsules_num, self.capsules_num)
        capsules_mask = tf.expand_dims(tf.range(self.capsules_num, dtype=tf.float32) + 1, 0) > tf.expand_dims(
            valid_capsules_num, 1)

        X_cate = []
        for feature in self.user_and_context_categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_cate.append(feature_tensor)
        X_cate = tf.concat(X_cate, axis=1)
        profile_output = tf.keras.layers.Flatten()(
            self.embed(X_cate))  # (batch_size,user_and_context_feature_cols*embedding_dim)
        # 和兴趣胶囊拼接的用户特征，限制一下维度
        profile_output = self.context_mlp(profile_output)  # (batch_size,embedding_dims//2)
        profile_output = self.context_bn(profile_output)

        profile_output = tf.expand_dims(profile_output, 1)  # 添加新的维度，形状变为 [batch_size, 1, embedding_dims//2]
        profile_output = tf.tile(profile_output,
                                 [1, tf.shape(interest_capsules)[1], 1])  # 在第一维上扩展到与 interest_capsules 相同的维度
        concated_capsules = tf.concat([profile_output, interest_capsules], axis=2)  # 在第二个维度上进行拼接

        mlped_capsules = self.final_mlp(concated_capsules)

        return mlped_capsules, capsules_mask, valid_capsules_num

    def save_interest_capsules(self, inputs):
        concated_capsules, capsules_mask, valid_capsules_num = self.generate_interest_capsules(
            inputs)  # (batch_size,capsule_num,capsule_dim)
        uids = inputs['uid']
        for idx, (uid, valid_capsules_num) in enumerate(zip(uids, valid_capsules_num)):
            valid_capsules = concated_capsules[idx, :tf.cast(valid_capsules_num, dtype=tf.int32), :]
            for valid_capsule in valid_capsules:
                # 改成写入redis的逻辑
                print(uid, valid_capsule)

    @staticmethod
    def compute_loss(inner_product):
        batch_size, num_samples = tf.shape(inner_product)[0], tf.shape(inner_product)[1]
        # 创建标签，第一个位置为1，其余位置为0
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, num_samples - 1))], axis=1)

        # 计算 softmax cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=inner_product)

        return loss

    def call(self, inputs):
        mlped_capsules, capsules_mask, valid_capsules_num = self.generate_interest_capsules(
            inputs)  # (batch_size,capsule_num,capsule_dim)

        X_item = []
        for feature in self.item_categorical_features:
            # (batch_size,num_samples)
            X_item.append(inputs[feature])
        X_item = tf.stack(X_item, axis=2)
        X_item = self.embed(X_item)
        X_item = tf.reshape(X_item, [-1, X_item.shape[1], X_item.shape[2] * X_item.shape[3]])

        # mlped_capsules:(batch_size,capsules_num,capsules_dim)
        # capsules_mask:(batch_size,capsules_num)
        # X_item:(batch_size,num_samples,capsules_dim)
        attentioned_user_interest = self.LAA_layer(mlped_capsules, capsules_mask, X_item)
        # (batch_size,num_samples,capsules_dim)

        inner_product = tf.reduce_sum(attentioned_user_interest * X_item, axis=-1)
        loss = self.compute_loss(inner_product)

        result = {'output': inner_product, 'loss': loss}
        return result


class AttnNet(tf.keras.layers.Layer):
    # 保证输入qkv的维度一致
    def __init__(self, **kwargs):
        super(AttnNet, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        q, k, v = inputs

        scores = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)  # scale factor
        scaled_scores = scores / tf.math.sqrt(dk)

        if mask is not None:
            mask = tf.expand_dims(mask, axis=1)
            # add a large negative value to the masked positions to make their softmax scores close to 0
            scaled_scores += (tf.cast(~mask, dtype=float) * -1e9)

        weights = tf.nn.softmax(scaled_scores, axis=-1)  # (..., seq_len_q, seq_len_k)
        output = tf.matmul(weights, v)  # (..., seq_len_q, depth_v)

        return output  # return the context vectors


class ShortTermInterestExtractor(tf.keras.layers.Layer):
    """
    sequence_lengths = tf.constant([7, 8, 9])  # 有效长度

    # 创建一个掩码矩阵，大小为最大长度*最大长度
    mask = tf.sequence_mask(sequence_lengths, 10, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.bool)

    short_term_layer = ShortTermInterestExtractor(lstm_units=64, lstm_layers=3, num_heads=8, attention_dim=128)
    user_embedding = tf.random.normal(shape=(3, 1, 64))
    shortterm_behaviors = tf.random.normal(shape=(3, 10, 64))  # batch_size=32, seq_len=10, emb_dim=64

    # Forward pass
    outputs = short_term_layer((user_embedding, shortterm_behaviors), mask)
    print(outputs.shape)
    """

    def __init__(self, lstm_units, lstm_layers, num_heads, attention_dim, **kwargs):
        super().__init__(**kwargs)
        self.lstm_layers = lstm_layers
        self.lstm_units = lstm_units
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=attention_dim)
        self.attn_net = AttnNet()
        self.lstm_layers_list = []
        for _ in range(lstm_layers):
            self.lstm_layers_list.append(tf.keras.layers.LSTM(lstm_units, return_sequences=True))

    def call(self, inputs, mask=None):
        user_embedding, short_term_interest = inputs

        for i in range(self.lstm_layers):
            short_term_interest = self.lstm_layers_list[i](short_term_interest, mask=mask)

        sequence_lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)

        # 创建一个掩码矩阵，大小为最大长度*最大长度
        mask_matrix = tf.sequence_mask(sequence_lengths, mask.shape[1], dtype=tf.float32)
        mask_matrix = tf.expand_dims(mask_matrix, -1)  # expand one dimension

        # Create a tensor with size batch_size*max_length*max_length,
        # where each sample's mask is a valid_len*valid_len rectangle
        attention_mask = mask_matrix * tf.transpose(mask_matrix, perm=[0, 2, 1])

        # Convert the mask back to boolean type
        attention_mask = tf.cast(attention_mask, dtype=tf.bool)

        short_term_interest = self.mha(short_term_interest, short_term_interest, short_term_interest,
                                       attention_mask=attention_mask)
        short_term_interest = self.attn_net([user_embedding, short_term_interest, short_term_interest], mask)
        short_term_interest = tf.squeeze(short_term_interest, axis=1)
        return short_term_interest


class LongTermInterestExtractor(tf.keras.layers.Layer):
    """
    long_term_layer = LongTermInterestExtractor(attention_dim=64)

    user_embedding=tf.random.normal(shape=(3,1,64))
    longterm_behaviors = [tf.random.normal(shape=(3, 10, 64)) for _ in range(3)]  # 3 feature sequences, batch_size=3, seq_len=10, emb_dim=64

    sequence_lengths = tf.constant([7, 8, 9])  # 有效长度
    mask = tf.sequence_mask(sequence_lengths, 10, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.bool)

    # Forward pass
    outputs = long_term_layer((user_embedding,longterm_behaviors), mask)
    print(outputs.shape)
    """

    def __init__(self, attention_dim, feature_col_num, **kwargs):
        super().__init__(**kwargs)
        self.attn_net = AttnNet()
        self.dnn = tf.keras.layers.Dense(feature_col_num * attention_dim, activation='tanh')

    def call(self, inputs, mask=None):
        user_embedding, long_term_interests = inputs
        attn_outputs = []
        for long_term_interest in long_term_interests:
            attn_output = self.attn_net([user_embedding, long_term_interest, long_term_interest], mask)
            attn_outputs.append(attn_output)

        concat_output = tf.concat(attn_outputs, axis=-1)
        long_term_interest = self.dnn(concat_output)
        long_term_interest = tf.squeeze(long_term_interest, axis=1)
        return long_term_interest


class FusionGate(tf.keras.layers.Layer):
    """
    fusion_layer = FusionGate(vector_dim=64)

    long_term_interest = tf.random.normal(shape=(3, 64))
    short_term_interest = tf.random.normal(shape=(3, 64))
    user_embedding=tf.random.normal(shape=(3,64))

    # Forward pass
    outputs = fusion_layer([user_embedding, long_term_interest, short_term_interest])
    print(outputs.shape)
    """

    def __init__(self, vector_dim, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(vector_dim)
        self.dense2 = tf.keras.layers.Dense(vector_dim)
        self.dense3 = tf.keras.layers.Dense(vector_dim)
        self.bias = self.add_weight(shape=(vector_dim,), initializer="zeros")

    def call(self, inputs):
        user_embedding, long_term_interest, short_term_interest = inputs
        gate_value = tf.nn.sigmoid(
            self.dense1(user_embedding) +
            self.dense2(long_term_interest) +
            self.dense3(short_term_interest) +
            self.bias[None, :]
        )
        output = (1 - gate_value) * long_term_interest + gate_value * short_term_interest
        return output


class SDMLayer(tf.keras.layers.Layer):
    def __init__(self, user_and_context_categorical_features=['uid', 'utag1', 'utag2', 'utag3', 'utag4'],
                 item_categorical_features=['label_goods_ids', 'label_shop_ids', 'label_cate_ids'],
                 longterm_series_features=['longterm_goods_ids', 'longterm_shop_ids', 'longterm_cate_ids'],
                 shortterm_series_features=['shortterm_goods_ids', 'shortterm_shop_ids', 'shortterm_cate_ids'],
                 feature_dims=160000, embedding_dims=16,
                 padding_index=0, lstm_layers=3, num_heads=8):
        super(SDMLayer, self).__init__()
        self.user_and_context_categorical_features = user_and_context_categorical_features
        self.item_categorical_features = item_categorical_features
        self.longterm_series_features = longterm_series_features
        self.shortterm_series_features = shortterm_series_features
        self.feature_dims = feature_dims
        self.embedding_dims = embedding_dims

        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims)
        self.padding_index = padding_index

        self.user_to_short = tf.keras.layers.Dense(len(shortterm_series_features) * embedding_dims, activation='tanh')
        self.user_to_long = tf.keras.layers.Dense(embedding_dims, activation='tanh')

        self.short_term_layer = ShortTermInterestExtractor(lstm_units=len(shortterm_series_features) * embedding_dims,
                                                           lstm_layers=lstm_layers, num_heads=num_heads,
                                                           attention_dim=embedding_dims)
        self.long_term_layer = LongTermInterestExtractor(attention_dim=embedding_dims,
                                                         feature_col_num=len(longterm_series_features))
        self.fusion_layer = FusionGate(vector_dim=len(shortterm_series_features) * embedding_dims)

    def build(self, input_shape):
        super().build(input_shape)

    def generate_interest_capsules(self, inputs):
        # 确定batch中每个样本行为序列的mask
        shortterm_mask = inputs[self.shortterm_series_features[0]] == self.padding_index
        longterm_mask = inputs[self.longterm_series_features[0]] == self.padding_index

        X_user = []
        for feature in self.user_and_context_categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_user.append(feature_tensor)
        X_user = tf.concat(X_user, axis=1)
        X_user = tf.keras.layers.Flatten()(self.embed(X_user))
        X_user_short = self.user_to_short(X_user)

        X_short = []
        for feature in self.shortterm_series_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_short.append(feature_tensor)
        X_short = tf.stack(X_short, axis=2)
        X_short = self.embed(X_short)
        X_short = tf.reshape(X_short, [-1, X_short.shape[1], X_short.shape[2] * X_short.shape[3]])
        short_term_interest = self.short_term_layer((tf.expand_dims(X_user_short, axis=1), X_short), shortterm_mask)

        longterm_behaviors = [self.embed(inputs[feature]) for feature in self.longterm_series_features]
        X_user_long = self.user_to_long(X_user)
        long_term_interest = self.long_term_layer((tf.expand_dims(X_user_long, axis=1), longterm_behaviors),
                                                  longterm_mask)

        user_embedding = self.fusion_layer([X_user_short, long_term_interest, short_term_interest])
        return user_embedding

    @staticmethod
    def compute_loss(inner_product):
        batch_size, num_samples = tf.shape(inner_product)[0], tf.shape(inner_product)[1]
        # 创建标签，第一个位置为1，其余位置为0
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, num_samples - 1))], axis=1)

        # 计算 softmax cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=inner_product)

        return loss

    def call(self, inputs):
        # 序列特征和候选商品特征
        X_item = []
        for feature in self.item_categorical_features:
            # (batch_size,num_samples)
            X_item.append(inputs[feature])
        X_item = tf.stack(X_item, axis=2)
        X_item = self.embed(X_item)
        X_item = tf.reshape(X_item, [-1, X_item.shape[1], X_item.shape[2] * X_item.shape[3]])

        user_embedding = self.generate_interest_capsules(inputs)
        user_embedding = tf.tile(tf.expand_dims(user_embedding, axis=1), [1, X_item.shape[1], 1])

        inner_product = tf.reduce_sum(user_embedding * X_item, axis=-1)

        loss = self.compute_loss(inner_product)

        result = {'output': inner_product, 'loss': loss}
        return result


class ComiRecDynamicRoutingLayer(tf.keras.layers.Layer):
    """
    layer=ComiRecDynamicRoutingLayer(4,6)
    batch_size,sequence_length,embedding_dims=3,5,8
    behaviors=tf.random.uniform(shape=[batch_size,sequence_length,embedding_dims], minval=0, maxval=1, dtype=tf.float32)
    sequence_mask = tf.random.uniform(shape=[batch_size,sequence_length], minval=0, maxval=2, dtype=tf.int32)
    sequence_mask = tf.cast(sequence_mask, dtype=tf.bool)
    print(layer((behaviors, sequence_mask)))
    """

    def __init__(self, capsules_num, capsules_dim, routing_rounds=3):
        super(ComiRecDynamicRoutingLayer, self).__init__()
        self.capsules_num = capsules_num
        self.capsules_dim = capsules_dim
        self.routing_rounds = routing_rounds

    def build(self, input_shape):
        super(ComiRecDynamicRoutingLayer, self).build(input_shape)
        sequence_length, embedding_dims = input_shape[0][-2], input_shape[0][-1]
        self.W = self.add_weight(shape=(self.capsules_num, sequence_length, embedding_dims, self.capsules_dim),
                                 initializer='random_normal',
                                 trainable=True)
        # W是连接输入的矩阵，把输入维度转换成胶囊维度
        # 和MIND不同，ComiRec每个胶囊的W输入输出都不是共享的
        self.B = self.add_weight(shape=(self.capsules_num, sequence_length),
                                 initializer='zeros',
                                 trainable=False)

    def squash(self, caps):
        n = tf.norm(caps, axis=2, keepdims=True)
        n_square = tf.square(n)

        return (n_square / ((1 + n_square) * n)) * caps

    def call(self, inputs):
        behaviors, valid_mask = inputs  # behaviors: (batch_size, sequence_length, embedding_dims)

        # Tile B for each sample in the batch
        B = tf.tile(self.B[None, :, :],
                    [tf.shape(behaviors)[0], 1, 1])  # B: (batch_size, capsules_num, sequence_length)

        # Mask padding indices' routing logit
        mask = tf.tile(valid_mask[:, None, :],
                       [1, self.capsules_num, 1])  # mask: (batch_size, capsules_num, sequence_length)
        drop = tf.ones_like(mask, dtype=tf.float32) * tf.float32.min

        # Transform behaviors into capsules dim on from each sequence position to each capsule respectively
        # behaviors : (batch_size,sequence_length,embedding_dims)
        # self.W :    (self.capsules_num,sequence_length,embedding_dims, self.capsules_dim)
        # 先expand_dims对齐 batch_size,sequence_length,capsules_num三维，然后矩阵乘embedding_dims-->capsules_dim
        # output :    (batch_size, capsules_num,sequence_length, capsule_dim)
        behaviors = tf.einsum('bse,nsec->bnsc', behaviors, self.W)

        # Perform routing rounds
        for i in range(self.routing_rounds):
            B_masked = tf.where(mask, B, drop)  # B_masked: (batch_size, capsules_num, sequence_length)
            C = tf.nn.softmax(B_masked, axis=2)  # W: (batch_size, capsules_num, sequence_length)

            if i < self.routing_rounds - 1:
                caps = tf.einsum('bns,bnsc->bnc', C, behaviors)  # caps: (batch_size, capsules_num, capsule_dims)
                caps = self.squash(caps)
                B += tf.einsum('bnc,bnsc->bns', caps, behaviors)  # B: (batch_size, capsules_num, sequence_length)
            else:
                caps = tf.einsum('bns,bnsc->bnc', C, behaviors)  # caps: (batch_size, capsules_num, capsule_dims)
                caps = self.squash(caps)

        return caps  # caps: (batch_size, capsules_num, capsule_dims)


class ComiRecSelfAttentiveLayer(tf.keras.layers.Layer):
    """
    layer=ComiRecSelfAttentiveLayer(4,6)
    batch_size,sequence_length,embedding_dims=3,5,8
    behaviors=tf.random.uniform(shape=[batch_size,sequence_length,embedding_dims], minval=0, maxval=1, dtype=tf.float32)
    sequence_mask = tf.random.uniform(shape=[batch_size,sequence_length], minval=0, maxval=2, dtype=tf.int32)
    sequence_mask = tf.cast(sequence_mask, dtype=tf.bool)
    print(layer((behaviors, sequence_mask)))
    """

    def __init__(self, capsules_num, capsules_dim, add_pos_emb=True):
        super(ComiRecSelfAttentiveLayer, self).__init__()
        self.capsules_num = capsules_num
        self.capsules_dim = capsules_dim
        self.add_pos_emb = add_pos_emb

    def build(self, input_shape):
        super(ComiRecSelfAttentiveLayer, self).build(input_shape)
        sequence_length, embedding_dims = input_shape[0][-2], input_shape[0][-1]
        self.W1 = self.add_weight(shape=(embedding_dims, self.capsules_dim),
                                  initializer='glorot_normal',
                                  trainable=True)

        self.W2 = self.add_weight(shape=(self.capsules_num, self.capsules_dim),
                                  initializer='glorot_normal',
                                  trainable=True)

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            tf.range(position)[:, tf.newaxis],
            tf.range(d_model)[tf.newaxis, :],
            d_model
        )
        # 将 sin 应用于数组中的偶数索引（indices 0, 2, ...）
        angle_rads = tf.where(tf.range(d_model) % 2 == 0,
                              tf.math.sin(angle_rads),
                              tf.math.cos(angle_rads))
        pos_encoding = angle_rads[tf.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        pos = tf.cast(pos, tf.float32)  # 将pos转换为tf.float32
        i = tf.cast(i, tf.float32)
        angle_rates = 1 / tf.math.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def call(self, inputs):
        behaviors, valid_mask = inputs  # behaviors: (batch_size, sequence_length, embedding_dims)

        if self.add_pos_emb:
            seq_length = tf.shape(behaviors)[1]
            pos_emb = self.positional_encoding(seq_length, behaviors.shape[-1])
            behaviors += pos_emb

        # Mask padding indices' routing logit
        mask = tf.tile(valid_mask[:, None, :],
                       [1, self.capsules_num, 1])  # mask: (batch_size, capsules_num, sequence_length)
        drop = tf.ones_like(mask, dtype=tf.float32) * tf.float32.min

        behaviors = tf.einsum('bse,ec->bsc', behaviors, self.W1)
        mids = tf.keras.activations.tanh(behaviors)
        attention_score = tf.einsum('bsc,nc->bns', mids, self.W2)
        attention_score = tf.where(mask, attention_score, drop)
        attention_score = tf.keras.activations.softmax(attention_score, axis=-1)
        # 论文里加权的是原始behaviors，这里改成加权转换成capsules_dim的向量，以和动态路由的输出形状保持一致
        caps = tf.einsum('bsc,bns->bnc', mids, attention_score)

        return caps  # caps: (batch_size, capsules_num, capsule_dims)


class ComiRecLayer(tf.keras.layers.Layer):
    """
    inputs = {'uids': tf.constant([0, 1, 2]),
        'visited_goods_ids': tf.constant([[30, 31, 32, 70, 71, 72, 73, 0, 0, 0], [33, 34, 35, 36, 81, 82, 83, 84, 0, 0],
                                          [37, 38, 39, 40, 41, 91, 92, 93, 94, 0]]),
        'label_goods_ids': tf.constant([[31, 32, 33, 34, 35, 36], [37, 38, 39, 40, 41, 42], [43, 44, 45, 46, 47, 48]]),
        'candidates': tf.reshape(tf.range(30), (3, 10)),
        'candidate_cates': tf.random.uniform((3, 10), minval=0, maxval=3, dtype=tf.int32)}
    layer = ComiRecLayer(embedding_dims=8, mode='DR', negative_sample_type='auto')
    print(layer.greedy_search_inference_parallel(inputs,6))
    """
    def __init__(self, item_categorical_features=['label_goods_ids', 'label_shop_ids', 'label_cate_ids'],
                 behavior_series_features=['visited_goods_ids', 'visited_shop_ids', 'visited_cate_ids'],
                 feature_dims=160000, embedding_dims=16,
                 activation='ReLU', padding_index=0, capsules_num=4, mode='DR', routing_rounds=3, add_pos_emb=True,
                 negative_sample_type='manual'):
        super(ComiRecLayer, self).__init__()
        if negative_sample_type == 'auto':
            item_categorical_features = item_categorical_features[0:1]
            behavior_series_features = behavior_series_features[0:1]
        self.item_categorical_features = item_categorical_features
        assert len(item_categorical_features) == len(
            behavior_series_features), "Features to be interacted should match in item and behavior series"
        self.behavior_series_features = behavior_series_features
        self.feature_dims = feature_dims
        self.embedding_dims = embedding_dims
        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims)
        # 也可以设定成某个超参数,这里直接取和拼接的embedding一样
        capsules_dim = embedding_dims * len(item_categorical_features)
        self.padding_index = padding_index

        if mode == 'DR':
            self.interest_extractor = ComiRecDynamicRoutingLayer(capsules_num, capsules_dim, routing_rounds)
        elif mode == 'SA':
            self.interest_extractor = ComiRecSelfAttentiveLayer(capsules_num, capsules_dim, add_pos_emb)
        else:
            raise ValueError("mode must be in ('DR','SA')")

        self.negative_sample_type = negative_sample_type

    def build(self, input_shape):
        super().build(input_shape)

    def generate_interest_capsules(self, inputs):
        # 确定batch中每个样本行为序列的mask
        valid_mask = inputs[self.behavior_series_features[0]] == self.padding_index

        X_series = tf.stack([inputs[feature] for feature in self.behavior_series_features],
                            axis=2)  # (batch_size, seq_length, feature_cols)
        batch_size, seq_length, feature_cols = X_series.shape
        X_series = tf.reshape(X_series, [-1, feature_cols * seq_length])  # (batch_size,feature_cols*seq_length)
        X_series = self.embed(X_series)  # (batch_size,feature_cols*seq_length,emb_dim)
        X_series = tf.reshape(X_series, [-1, seq_length,
                                         feature_cols * self.embedding_dims])  # (batch_size,seq_length,feature_cols*emb_dim)

        interest_capsules = self.interest_extractor((X_series, valid_mask))
        return interest_capsules

    def save_interest_capsules(self, inputs):
        interest_capsules = self.generate_interest_capsules(inputs)  # (batch_size,capsule_num,capsule_dim)
        uids = inputs['uid']
        for idx, (uid, user_interests) in enumerate(zip(uids, interest_capsules)):
            for user_interest in user_interests:
                # 改成写入redis或者索引的逻辑
                print(uid, user_interest)

    @staticmethod
    def compute_loss(inner_product):
        batch_size, num_samples = tf.shape(inner_product)[0], tf.shape(inner_product)[1]
        # 创建标签，第一个位置为1，其余位置为0
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, num_samples - 1))], axis=1)

        # 计算 softmax cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=inner_product)

        return loss

    def call(self, inputs):
        if self.negative_sample_type == 'auto':
            return self.automatically_negative_sampled_call(inputs)
        elif self.negative_sample_type == 'manual':
            return self.manually_negative_sampled_call(inputs)
        else:
            raise ValueError("negative_sample_type must be in ('auto','manual')")

    def manually_negative_sampled_call(self, inputs):
        """
        inputs = {
        'visited_goods_ids': tf.constant([[30, 31, 32, 70, 71, 72, 73, 0, 0, 0], [33, 34, 35, 36, 81, 82, 83, 84, 0, 0],
                                          [37, 38, 39, 40, 41, 91, 92, 93, 94, 0]]),
        'visited_shop_ids': tf.constant([[42, 43, 44, 70, 71, 72, 73, 0, 0, 0], [45, 46, 47, 48, 81, 82, 83, 84, 0, 0],
                                         [49, 50, 51, 52, 53, 91, 92, 93, 94, 0]]),
        'visited_cate_ids': tf.constant([[54, 55, 56, 70, 71, 72, 73, 0, 0, 0], [57, 58, 59, 60, 81, 82, 83, 84, 0, 0],
                                         [61, 62, 63, 64, 65, 91, 92, 93, 94, 0]]),
        'label_cate_ids': tf.constant([[11, 12, 13, 14, 15, 16], [17, 18, 19, 20, 21, 22], [23, 24, 25, 26, 27, 28]]),
        'label_goods_ids': tf.constant([[31, 32, 33, 34, 35, 36], [37, 38, 39, 40, 41, 42], [43, 44, 45, 46, 47, 48]]),
        'label_shop_ids': tf.constant([[51, 52, 53, 54, 55, 56], [57, 58, 59, 60, 61, 62], [63, 64, 65, 66, 67, 68]])
    }
        """
        interest_capsules = self.generate_interest_capsules(inputs)  # (batch_size,capsule_num,capsule_dim)
        batch_size = tf.shape(interest_capsules)[0]

        X_item = []
        for feature in self.item_categorical_features:
            # (batch_size,num_samples)
            X_item.append(inputs[feature])
        X_item = tf.stack(X_item, axis=2)
        X_item = self.embed(X_item)
        # (batch_size,num_samples,capsule_dim)
        X_item = tf.reshape(X_item, [-1, X_item.shape[1], X_item.shape[2] * X_item.shape[3]])

        print(X_item.shape)
        print(interest_capsules.shape)

        # 由于前面的设定，capsule_dim和X_item维度是相等的
        # interest_capsules (batch_size,capsule_num,capsule_dim)
        # X_item            (batch_size,num_samples,capsule_dim)
        capsule_num = tf.shape(interest_capsules)[1]
        num_samples = tf.shape(X_item)[1]
        interest_capsules_expanded = tf.tile(tf.expand_dims(interest_capsules, axis=1), [1, num_samples, 1, 1])
        X_item_expanded = tf.tile(tf.expand_dims(X_item, axis=2), [1, 1, capsule_num, 1])
        # (batch_size,num_samples,capsule_num)
        interest_product = tf.reduce_sum(interest_capsules_expanded * X_item_expanded, axis=-1)
        chosen_capsule = tf.argmax(interest_product, axis=-1)
        chosen_capsule = tf.cast(chosen_capsule, dtype=tf.int32)

        # # 使用 tf.gather_nd 提取 capsules
        batch_indices = tf.reshape(tf.range(0, batch_size), [batch_size, 1, 1])
        batch_indices = tf.tile(batch_indices, [1, num_samples, 1])
        num_samples_indices = tf.reshape(tf.range(0, num_samples), [1, num_samples, 1])
        num_samples_indices = tf.tile(num_samples_indices, [batch_size, 1, 1])
        # 合并 batch_indices、num_samples_indices 和 chosen_capsule 以创建 indices
        indices = tf.concat([batch_indices, num_samples_indices, tf.expand_dims(chosen_capsule, -1)], axis=-1)
        # (batch_size,num_samples,capsule_dim)
        selected_capsules = tf.gather_nd(interest_capsules_expanded, indices)
        # X_item  (batch_size,num_samples,capsule_dim)

        inner_product = tf.reduce_sum(selected_capsules * X_item, axis=-1)
        loss = self.compute_loss(inner_product)

        result = {'output': selected_capsules, 'loss': loss}
        return result

    def automatically_negative_sampled_call(self, inputs):
        """
        直接使用tf.nn.sampled_softmax_loss的话只能用id类embedding
        layer=ComiRecLayer(item_categorical_features=['label_goods_ids'],
                 behavior_series_features=['visited_goods_ids'])
        inputs = {
        'visited_goods_ids': tf.constant([[30, 31, 32, 70, 71, 72, 73, 0, 0, 0], [33, 34, 35, 36, 81, 82, 83, 84, 0, 0],
                                          [37, 38, 39, 40, 41, 91, 92, 93, 94, 0]]),
        'label_goods_ids': tf.constant([1, 2, 3]),
    }
        """
        interest_capsules = self.generate_interest_capsules(inputs)  # (batch_size,capsule_num,capsule_dim)
        batch_size = tf.shape(interest_capsules)[0]

        X_item = []
        for feature in self.item_categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_item.append(feature_tensor)
        X_item = tf.concat(X_item, axis=1)
        X_item = tf.keras.layers.Flatten()(self.embed(X_item))

        # 由于前面的设定，capsule_dim和X_item维度是相等的
        # interest_capsules (batch_size,capsule_num,capsule_dim)
        # X_item            (batch_size,capsule_dim)
        capsule_num = tf.shape(interest_capsules)[1]
        X_item_expanded = tf.tile(tf.expand_dims(X_item, axis=1), [1, capsule_num, 1])
        # (batch_size,capsule_num)
        interest_product = tf.reduce_sum(interest_capsules * X_item_expanded, axis=-1)
        # (batch_size,)
        chosen_capsule = tf.argmax(interest_product, axis=-1)
        chosen_capsule = tf.cast(chosen_capsule, dtype=tf.int32)
        # 为chosen_capsule创建一个批处理索引
        batch_indices = tf.range(batch_size)
        indices = tf.stack([batch_indices, chosen_capsule], axis=1)

        # 使用tf.gather_nd提取对应的capsule_dim
        # (batch_size,capsule_dim)
        selected_capsules = tf.gather_nd(interest_capsules, indices)

        labels = tf.reshape(tf.cast(inputs['label_goods_ids'], tf.int32), [-1, 1])
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(
                weights=self.embed.weights[0],
                biases=tf.zeros([self.feature_dims]),
                labels=labels,
                inputs=selected_capsules,
                num_sampled=30,
                num_classes=self.feature_dims
            )
        )

        result = {'output': selected_capsules, 'loss': loss}
        return result

    def greedy_search_inference(self, inputs, k,lambda_1=0.5):
        """
        inputs = {
         'uids': tf.constant([0, 1, 2]),
         'visited_goods_ids': tf.constant([[30, 31, 32, 70, 71, 72, 73, 0, 0, 0], [33, 34, 35, 36, 81, 82, 83, 84, 0, 0],
                                           [37, 38, 39, 40, 41, 91, 92, 93, 94, 0]]),
         'label_goods_ids': tf.constant([[31, 32, 33, 34, 35, 36], [37, 38, 39, 40, 41, 42], [43, 44, 45, 46, 47, 48]]),
         'candidates':tf.reshape(tf.range(30), (3,10)),
         'candidate_cates':tf.random.uniform((3,10), minval=0, maxval=3, dtype=tf.int32)}
        layer = ComiRecLayer(embedding_dims=8, mode='DR', negative_sample_type='auto')
     }
         """
        candidates = inputs.pop('candidates')
        candidate_cates = inputs.pop('candidate_cates')
        uids=inputs['uids']
        interest_capsules = self.generate_interest_capsules(inputs)  # (batch_size,num_capsules,capsule_dim)
        X_candidates=self.embed(candidates)                                    # (batch_size,candidates,capsule_dim)
        inner_products=tf.einsum('bnd,bcd->bcn',interest_capsules,X_candidates)  # (batch_size,candidates,num_capsules)
        best_products=tf.reduce_max(inner_products,axis=-1)                      # (batch_size,candidates)

        uids,candidates,best_products,candidate_cates=map(lambda x:x.numpy().tolist(),[uids,candidates,best_products,candidate_cates])
        result= {}
        for uid,candidate,best_product,candidate_cate in zip(uids,candidates,best_products,candidate_cates):
            selected_candidates=[]
            selected_cates={cate:0 for cate in candidate_cate}
            for round in range(k):
                best_score = -np.inf
                best_idx = None
                for idx,(item,cate,product) in enumerate(zip(candidate,candidate_cate,best_product)):
                    cate_score=len(selected_candidates)-selected_cates[cate]
                    score=product+lambda_1*cate_score
                    if score>best_score:
                        best_score=score
                        best_idx=idx
                selected_candidates.append(candidate[best_idx])
                selected_cates[candidate_cate[best_idx]] += 1
                del candidate[best_idx]
                del candidate_cate[best_idx]
                del best_product[best_idx]
            result[uid]=selected_candidates
        return result


    def greedy_search_inference_parallel(self, inputs, k, lambda_1=0.5):
        candidates = inputs.pop('candidates')
        candidate_cates = inputs.pop('candidate_cates')
        interest_capsules = self.generate_interest_capsules(inputs)
        X_candidates = self.embed(candidates)
        inner_products = tf.einsum('bnd,bcd->bcn', interest_capsules, X_candidates)
        initial_scores = tf.reduce_max(inner_products, axis=-1)

        flattened_tensor = tf.reshape(candidate_cates, [-1])
        unique_values, _ = tf.unique(flattened_tensor)
        num_unique_cates = tf.shape(unique_values)[0]

        selected_candidates_tensor = tf.zeros((tf.shape(inputs['uids'])[0], k), dtype=tf.int32)

        candidates=tf.concat([-1*tf.zeros([tf.shape(candidates)[0], 1], dtype=candidates.dtype), candidates], axis=1)
        candidate_cates = tf.concat([(num_unique_cates) * tf.ones([tf.shape(candidate_cates)[0], 1], dtype=candidate_cates.dtype), candidate_cates],
                               axis=1)
        initial_scores = tf.concat([-np.inf * tf.ones([tf.shape(initial_scores)[0], 1], dtype=initial_scores.dtype), initial_scores],
                               axis=1)

        def loop_condition(i, selected_candidates_tensor, initial_scores):
            return i < k

        def loop_body(i, selected_candidates_tensor, initial_scores):
            current_selected_cates = tf.gather(candidate_cates, selected_candidates_tensor, batch_dims=1)
            cate_scores = k - tf.math.reduce_sum(
                tf.one_hot(current_selected_cates, depth=num_unique_cates+1), axis=1)
            combined_scores = initial_scores + lambda_1 * tf.gather(cate_scores, candidate_cates, batch_dims=1)
            top_candidate = tf.argmax(combined_scores, axis=-1, output_type=tf.int32)

            # Update selected_candidates_tensor
            batch_indices = tf.range(tf.shape(top_candidate)[0])
            indices_for_selected = tf.stack([batch_indices, tf.fill(tf.shape(batch_indices), i)], axis=1)
            selected_candidates_tensor = tf.tensor_scatter_nd_update(selected_candidates_tensor, indices_for_selected,
                                                                     top_candidate)
            # Set the scores of the selected candidates to -inf
            indices_for_scores = tf.stack([batch_indices, top_candidate], axis=1)
            updates = tf.ones_like(top_candidate, dtype=tf.float32) * -float('inf')
            initial_scores = tf.tensor_scatter_nd_update(initial_scores, indices_for_scores, updates)

            return i + 1, selected_candidates_tensor, initial_scores

        _, selected_candidates_final, _ = tf.while_loop(loop_condition, loop_body,
                                                        [0, selected_candidates_tensor, initial_scores])


        selected_candidates_final=tf.gather(candidates, selected_candidates_final, batch_dims=1)
        # Convert to dictionary format for consistency with original function
        uids_list, selected_candidates_list = inputs[
            'uids'].numpy().tolist(), selected_candidates_final.numpy().tolist()
        result = {uid: candidates for uid, candidates in zip(uids_list, selected_candidates_list)}

        return result


if __name__ == '__main__':
    """
    inputs = {
        'uid': tf.constant([0, 1, 2]),
        'utag1': tf.constant([6, 7, 8]),
        'utag2': tf.constant([9, 10, 11]),
        'utag3': tf.constant([12, 13, 14]),
        'utag4': tf.constant([15, 16, 17]),
        'visited_goods_ids': tf.constant([[30, 31, 32,70,71,72,73,0,0, 0], [33, 34, 35, 36,81,82,83,84, 0, 0], [37, 38, 39, 40, 41,91,92,93,94, 0]]),
        'visited_shop_ids': tf.constant([[42, 43, 44,70,71,72,73,0,0, 0], [45, 46, 47, 48,81,82,83,84, 0, 0], [49, 50, 51, 52, 53, 91,92,93,94, 0]]),
        'visited_cate_ids': tf.constant([[54, 55, 56,70,71,72,73,0,0, 0], [57, 58, 59, 60, 81,82,83,84,0, 0], [61, 62, 63, 64, 65, 91,92,93,94, 0]]),
        'label_cate_ids': tf.constant([[11, 12, 13,14,15,16],[17,18,19,20,21,22],[23,24,25,26,27,28]]),
        'label_goods_ids': tf.constant([[31, 32, 33,34,35,36],[37,38,39,40,41,42],[43,44,45,46,47,48]]),
        'label_shop_ids': tf.constant([[51, 52, 53,54,55,56],[57,58,59,60,61,62],[63,64,65,66,67,68]])
        # 训练和验证时，采用sampled_softmax,约定生成数据时，每一行总是第一个样本为正，其余样本为负，不需要再单独传label字段
        # 预测推理时，不需要有候选商品侧信息，所以也不存在标签
    }
    layer = MINDLayer(embedding_dims=8)
    input_dic = {}
    max_length=10
    #for feature in layer.user_and_context_categorical_features+ layer.item_categorical_features:
    #    input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.user_and_context_categorical_features:
        input_dic[feature] = tf.keras.Input(shape=(), name=feature, dtype=tf.int64)
    for feature in layer.item_categorical_features:
        input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.behavior_series_features:
        input_dic[feature] = tf.keras.Input(shape=(max_length,), name=feature, dtype=tf.int64)
    output = layer(input_dic)
    model = tf.keras.Model(input_dic, output)
    model.summary()
    print(model(inputs))
    """
    """
    layer = MultiInterestExtractorLayer(4)
    batch_size, sequence_length, embedding_dims = 3, 5, 8
    behaviors = tf.random.uniform(shape=[batch_size, sequence_length, embedding_dims], minval=0, maxval=1,
                                  dtype=tf.float32)
    sequence_mask = tf.random.uniform(shape=[batch_size, sequence_length], minval=0, maxval=2, dtype=tf.int32)
    sequence_mask = tf.cast(sequence_mask, dtype=tf.bool)
    print(layer((behaviors, sequence_mask)))
    """
    """
    coaction_unit = CoActionUnit(induction_feature_num=20,
                                 feed_feature_num=20,
                                 feed_feature_dim=5,
                                 layer_unit_coef=[1, 1, 1],
                                 order=3,
                                 order_independent=True)

    # 创建一些随机输入数据
    induction_feature = tf.constant([1, 2, 3])
    feed_feature = tf.constant([4, 5, 6])

    # 调用 coaction_unit
    output = coaction_unit([induction_feature, feed_feature])

    print(output)
    print('*******************')

    induction_feature = tf.constant([1, 2, 3])
    feed_feature = tf.constant([[4, 5, 0, 0], [8, 9, 10, 0], [12, 13, 14, 15]])

    # 调用 coaction_unit
    output = coaction_unit([induction_feature, feed_feature])

    # 打印输出
    print(output)
    """
    """
    inputs = {
        'uid': tf.constant([0, 1, 2]),
        'visited_goods_ids': tf.constant([[30, 31, 32, 70, 71, 72, 73, 0, 0, 0], [33, 34, 35, 36, 81, 82, 83, 84, 0, 0],
                                          [37, 38, 39, 40, 41, 91, 92, 93, 94, 0]]),
        'visited_shop_ids': tf.constant([[42, 43, 44, 70, 71, 72, 73, 0, 0, 0], [45, 46, 47, 48, 81, 82, 83, 84, 0, 0],
                                         [49, 50, 51, 52, 53, 91, 92, 93, 94, 0]]),
        'visited_cate_ids': tf.constant([[54, 55, 56, 70, 71, 72, 73, 0, 0, 0], [57, 58, 59, 60, 81, 82, 83, 84, 0, 0],
                                         [61, 62, 63, 64, 65, 91, 92, 93, 94, 0]]),
        'label_cate_ids': tf.constant([[11, 12, 13, 14, 15, 16], [17, 18, 19, 20, 21, 22], [23, 24, 25, 26, 27, 28]]),
        'label_goods_ids': tf.constant([[31, 32, 33, 34, 35, 36], [37, 38, 39, 40, 41, 42], [43, 44, 45, 46, 47, 48]]),
        'label_shop_ids': tf.constant([[51, 52, 53, 54, 55, 56], [57, 58, 59, 60, 61, 62], [63, 64, 65, 66, 67, 68]])
    }
    # auto时需要更新label_goods_ids值
    inputs['label_goods_ids'] = tf.constant([100, 101, 102])
    layer = ComiRecLayer(embedding_dims=8, mode='DR', negative_sample_type='auto')
    print(layer(inputs))
    # print('************************')
    input_dic = {}
    max_length = 10
    for feature in layer.item_categorical_features:
        input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.behavior_series_features:
        input_dic[feature] = tf.keras.Input(shape=(max_length,), name=feature, dtype=tf.int64)
    output = layer(input_dic)
    model = tf.keras.Model(input_dic, output)
    model.summary()
    print(model(inputs))
    """
    """
    inputs = {'uids': tf.constant([0, 1, 2]),
        'visited_goods_ids': tf.constant([[30, 31, 32, 70, 71, 72, 73, 0, 0, 0], [33, 34, 35, 36, 81, 82, 83, 84, 0, 0],
                                          [37, 38, 39, 40, 41, 91, 92, 93, 94, 0]]),
        'label_goods_ids': tf.constant([[31, 32, 33, 34, 35, 36], [37, 38, 39, 40, 41, 42], [43, 44, 45, 46, 47, 48]]),
        'candidates': tf.reshape(tf.range(30), (3, 10)),
        'candidate_cates': tf.random.uniform((3, 10), minval=0, maxval=3, dtype=tf.int32)}
    layer = ComiRecLayer(embedding_dims=8, mode='DR', negative_sample_type='auto')
    print(layer.greedy_search_inference_parallel(inputs,6))
    """
    """
    inputs = {
        'uid': tf.constant([0, 1, 2]),
        'utag1': tf.constant([6, 7, 8]),
        'utag2': tf.constant([9, 10, 11]),
        'utag3': tf.constant([12, 13, 14]),
        'utag4': tf.constant([15, 16, 17]),
        'longterm_goods_ids': tf.constant([[30, 31, 32, 70, 71, 72, 73, 0, 0, 0], [33, 34, 35, 36, 81, 82, 83, 84, 0, 0],
                                          [37, 38, 39, 40, 41, 91, 92, 93, 94, 0]]),
        'longterm_shop_ids': tf.constant([[42, 43, 44, 70, 71, 72, 73, 0, 0, 0], [45, 46, 47, 48, 81, 82, 83, 84, 0, 0],
                                         [49, 50, 51, 52, 53, 91, 92, 93, 94, 0]]),
        'longterm_cate_ids': tf.constant([[54, 55, 56, 70, 71, 72, 73, 0, 0, 0], [57, 58, 59, 60, 81, 82, 83, 84, 0, 0],
                                         [61, 62, 63, 64, 65, 91, 92, 93, 94, 0]]),
        'shortterm_goods_ids': tf.constant([[1,2,3,4, 0], [5,6,7, 0, 0],
                                          [8,9,10,11,12]]),
        'shortterm_shop_ids': tf.constant([[13,14,15,16,0], [17,18,19, 0, 0],
                                          [20,21,22,23,24]]),
        'shortterm_cate_ids': tf.constant([[25,26,27,28,0], [29,30,31, 0, 0],
                                          [32,33,34,35,36]]),
        'label_cate_ids': tf.constant([[11, 12, 13, 14, 15, 16], [17, 18, 19, 20, 21, 22], [23, 24, 25, 26, 27, 28]]),
        'label_goods_ids': tf.constant([[31, 32, 33, 34, 35, 36], [37, 38, 39, 40, 41, 42], [43, 44, 45, 46, 47, 48]]),
        'label_shop_ids': tf.constant([[51, 52, 53, 54, 55, 56], [57, 58, 59, 60, 61, 62], [63, 64, 65, 66, 67, 68]])
    }
    user_and_context_categorical_features = ['uid', 'utag1', 'utag2', 'utag3', 'utag4'],
    item_categorical_features = ['label_goods_ids', 'label_shop_ids', 'label_cate_ids'],
    longterm_series_features = ['longterm_goods_ids', 'longterm_shop_ids', 'longterm_cate_ids'],
    shortterm_series_features = ['shortterm_goods_ids', 'shortterm_shop_ids', 'shortterm_cate_ids']

    layer = SDMLayer()
    print(layer(inputs))
    print('****************')
    input_dict = {}
    shortterm_max_length = 5
    longterm_max_length = 10
    for feature in layer.user_and_context_categorical_features + layer.item_categorical_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.shortterm_series_features:
        input_dict[feature] = tf.keras.Input(shape=(shortterm_max_length,), name=feature, dtype=tf.int64)
    for feature in layer.longterm_series_features:
        input_dict[feature] = tf.keras.Input(shape=(longterm_max_length,), name=feature, dtype=tf.int64)
    output = layer(input_dict)
    model = tf.keras.Model(input_dict, output)
    model.summary()
    print(model(inputs))
    """
    sequence_lengths = tf.constant([7, 8, 9])  # 有效长度

    # 创建一个掩码矩阵，大小为最大长度*最大长度
    mask = tf.sequence_mask(sequence_lengths, 10, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.bool)

    sequence_lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)

    # 创建一个掩码矩阵，大小为最大长度*最大长度
    mask_matrix = tf.sequence_mask(sequence_lengths, mask.shape[1], dtype=tf.float32)
    mask_matrix = tf.expand_dims(mask_matrix, -1)  # expand one dimension

    # Create a tensor with size batch_size*max_length*max_length,
    # where each sample's mask is a valid_len*valid_len rectangle
    attention_mask = mask_matrix * tf.transpose(mask_matrix, perm=[0, 2, 1])

    # Convert the mask back to boolean type
    attention_mask = tf.cast(attention_mask, dtype=tf.bool)

    print(attention_mask)

