import tensorflow as tf
from tensorflow.python.ops.math_ops import MatMul
from tensorflow.python.ops.nn_ops import BiasAdd
from tensorflow.python.keras import activations, initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D
from tensorflow.keras.layers import Embedding,BatchNormalization,LayerNormalization,MultiHeadAttention,Dropout,Lambda,Activation
from tensorflow.keras.regularizers import l2

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


def make_mlp_layer(units, activation='PReLU', normalization='layernorm', softmax_units=-1, sigmoid_units=False, dropout_rate=0.0):
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
        if dropout_rate > 0:
            mlp.add(tf.keras.layers.Dropout(dropout_rate)) # 添加dropout层
    if softmax_units > 0:
        mlp.add(tf.keras.layers.Dense(softmax_units, activation='softmax'))
    elif sigmoid_units:
        mlp.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return mlp


def set_custom_initialization(layer, initializer=tf.keras.initializers.GlorotUniform()):
    if hasattr(layer, 'layers'):  # 如果层有子层
        for sublayer in layer.layers:
            set_custom_initialization(sublayer, initializer)
    if hasattr(layer, 'kernel_initializer'):
        layer.kernel_initializer = initializer
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            # 重新初始化权重
            new_kernel = initializer(shape=layer.kernel.shape)
            layer.kernel.assign(new_kernel)


class SDIMLayer(tf.keras.layers.Layer):
    # python字典的键不能是tensor，图模式下不能把tensor转成普通python对象
    def __init__(self, user_and_context_categorical_features=['uid', 'utag1', 'utag2', 'utag3', 'utag4'],
                 item_categorical_features=['label_goods_ids', 'label_shop_ids', 'label_cate_ids'],
                 longterm_series_features=['longterm_goods_ids', 'longterm_shop_ids', 'longterm_cate_ids'],
                 shortterm_series_features=['shortterm_goods_ids', 'shortterm_shop_ids', 'shortterm_cate_ids'],
                 feature_dims=160000, embedding_dims=16,
                 padding_index=0, num_heads=8,key_dim=4,
                 activation='ReLU',lsh_group_num=4,lsh_group_length=3):
        super(SDIMLayer, self).__init__()
        self.user_and_context_categorical_features = user_and_context_categorical_features
        self.item_categorical_features = item_categorical_features
        self.longterm_series_features = longterm_series_features
        self.shortterm_series_features = shortterm_series_features
        assert len(item_categorical_features)==len(longterm_series_features)==len(shortterm_series_features)
        self.feature_dims = feature_dims
        self.embedding_dims = embedding_dims

        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims)
        self.padding_index = padding_index

        self.extended_embedding_dims=len(item_categorical_features)*embedding_dims
        self.H = self.add_weight(
            name="projection_matrix",
            shape=(self.extended_embedding_dims, lsh_group_num,lsh_group_length),
            initializer='random_normal',
            trainable=False
        )

        self.lsh_group_num=lsh_group_num
        self.lsh_group_length=lsh_group_length
        self.m_values = tf.constant([2 ** i for i in range(lsh_group_length)], dtype=tf.int32)

        self.shortterm_mha=tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.longterm_mha=tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

        self.final_mlp=make_mlp_layer([200, 80], activation=activation, sigmoid_units=True)

        self.lsh_dict={}

    def build(self, input_shape):
        super().build(input_shape)

    def generate_longterm_interest(self, uids, item_embedding, longterm_series_embedding, longterm_mask,once=True,update=True):
        if once:
            # 计算 series_codes 并转换为二进制表示
            series_codes = tf.sign(tf.einsum('ble,egm->blgm', longterm_series_embedding, self.H))
            series_binary = tf.where(series_codes == 1, 1, 0)
            # 将二进制表示转换为整数
            series_int_codes = tf.tensordot(series_binary, self.m_values, axes=[[-1], [0]])  # blg
            # 创建一个mask，用于从 longterm_series_embedding 中选择有效的 embeddings
            mask = tf.cast(longterm_mask, tf.float32)[:, :, tf.newaxis]
            masked_series_embedding = longterm_series_embedding * mask  # ble
            # 使用 one_hot 编码表示每个组的整数代码
            max_code = 2 ** series_codes.shape[-1]
            one_hot_codes = tf.one_hot(series_int_codes, max_code, axis=-1)  # blgc
            # 生成 c 专属的 embedding
            c_specific_embedding = tf.einsum('blgc,ble->bgce', one_hot_codes, masked_series_embedding)  # bgce
            # 计算每个组的有效数量
            valid_count = tf.reduce_sum(one_hot_codes, axis=1)  # bgc
            division_result = c_specific_embedding / tf.expand_dims(valid_count, axis=-1)
            # 使用tf.where替换NaN和inf为0
            safe_result = tf.where(tf.math.is_finite(division_result), division_result, tf.zeros_like(division_result)) # bgce
            item_codes = tf.sign(tf.einsum('bse,egm->bsgm', item_embedding, self.H))
            items_binary = tf.where(item_codes == 1, 1, 0)
            item_int_codes = tf.tensordot(items_binary, self.m_values, axes=[[-1], [0]])  # bsg
            item_one_hot_codes = tf.one_hot(item_int_codes, max_code, axis=-1)  # bsgc
            candidate_interest=tf.einsum('bgce,bsgc->bsge',safe_result,item_one_hot_codes)
            candidate_interest=tf.reduce_mean(candidate_interest,axis=2) # bse

        else:
            if update:
                self.update_longterm_interest(uids, longterm_series_embedding, longterm_mask)
            candidate_interest=self.read_longterm_interest(uids, item_embedding, longterm_series_embedding)
        return candidate_interest

    def read_longterm_interest(self, uids, item_embedding, longterm_series_embedding):
        item_codes = tf.sign(tf.einsum('bse,egm->bsgm', item_embedding, self.H))
        items_binary = tf.where(item_codes == 1, 1, 0)
        item_int_codes = tf.tensordot(items_binary, self.m_values, axes=[[-1], [0]])  # bsg

        b_list = []
        for b in range(tf.shape(longterm_series_embedding)[0]):  # batch
            uid = uids[b].numpy().item()
            g_list = []
            for g in range(self.lsh_group_num):  # g
                values = item_int_codes[b, :, g]
                s_list = []
                for value in values:  # s
                    key = (uid, g, value.numpy().item())
                    embedding_sum, valid_cnt = self.lsh_dict.get(key, (tf.zeros(self.extended_embedding_dims), 0))
                    interest = embedding_sum / valid_cnt if valid_cnt > 0 else tf.zeros(self.extended_embedding_dims)
                    s_list.append(interest)
                g_list.append(s_list)
            b_list.append(g_list)
        candidate_interest = tf.convert_to_tensor(b_list)  # bgse
        candidate_interest = tf.reduce_mean(candidate_interest, axis=1)  # bse
        return candidate_interest

    def update_longterm_interest(self,uids,longterm_series_embedding, longterm_mask):
        # 计算 series_codes 并转换为二进制表示
        series_codes = tf.sign(tf.einsum('ble,egm->blgm', longterm_series_embedding, self.H))
        series_binary = tf.where(series_codes == 1, 1, 0)

        # 将二进制表示转换为整数
        series_int_codes = tf.tensordot(series_binary, self.m_values, axes=[[-1], [0]]) # blg

        # 创建一个mask，用于从 longterm_series_embedding 中选择有效的 embeddings
        mask = tf.cast(longterm_mask, tf.float32)[:, :, tf.newaxis]
        masked_series_embedding = longterm_series_embedding * mask     # ble

        # 使用 one_hot 编码表示每个组的整数代码
        max_code = 2 ** series_codes.shape[-1]
        one_hot_codes = tf.one_hot(series_int_codes, max_code, axis=-1)  # blgc

        # 生成 c 专属的 embedding
        c_specific_embedding = tf.einsum('blgc,ble->bgce', one_hot_codes, masked_series_embedding)  # bgce

        # 计算每个组的有效数量
        valid_count = tf.reduce_sum(one_hot_codes, axis=1)  # bgc

        for b in range(tf.shape(series_codes)[0]):  # batch
            uid = uids[b].numpy().item()
            for g in range(self.lsh_group_num):  # group
                unique_codes, _ = tf.unique(series_int_codes[b, :, g])  # 获取 g 组内唯一的整数代码
                for c in unique_codes:
                    c = c.numpy().item()
                    key = (uid, g, c)
                    if valid_count[b, g, c] > 0:  # 只添加计数不为0的组
                        if key in self.lsh_dict:
                            existing_embedding_sum, existing_count = self.lsh_dict[key]
                            self.lsh_dict[key] = (existing_embedding_sum + c_specific_embedding[b, g, c],
                                                existing_count + valid_count[b, g, c])
                        else:
                            self.lsh_dict[key] = (c_specific_embedding[b, g, c], valid_count[b, g, c])
        # self.lsh_dict的写入和更新，可以修改成其它写入过程，写入硬盘或内存
        return

    def generate_shorterm_interest(self,item_embedding,shortterm_series_embedding,shortterm_mask):
        sequence_lengths = tf.reduce_sum(tf.cast(~shortterm_mask, tf.int32), axis=1)

        # 创建一个掩码矩阵，大小为batch_size*最大长度
        mask_matrix = tf.sequence_mask(sequence_lengths, shortterm_mask.shape[1], dtype=tf.float32)
        attention_mask = tf.expand_dims(mask_matrix, 1)  # expand one dimension
        shortterm_interest = self.shortterm_mha(item_embedding, shortterm_series_embedding, shortterm_series_embedding,
                                   attention_mask=attention_mask)
        return shortterm_interest

    def stack_feature_embeddings(self,inputs,features):
        X = []
        for feature in features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X.append(feature_tensor)
        X = tf.stack(X, axis=2)
        X = self.embed(X)
        X = tf.reshape(X, [-1, X.shape[1], X.shape[2] * X.shape[3]])
        return X

    def call(self, inputs):
        uids=inputs['uid']
        shortterm_mask = inputs[self.shortterm_series_features[0]] == self.padding_index
        longterm_mask = inputs[self.longterm_series_features[0]] == self.padding_index

        X_item = self.stack_feature_embeddings(inputs,self.item_categorical_features)
        # X_item : (batch_size,item_candiates,embedding_dim*cols)
        X_user = self.stack_feature_embeddings(inputs,self.user_and_context_categorical_features)
        X_short = self.stack_feature_embeddings(inputs,self.shortterm_series_features)
        X_long=self.stack_feature_embeddings(inputs,self.longterm_series_features)

        shortterm_insterest=self.generate_shorterm_interest(X_item,X_short,shortterm_mask)  # bse
        longterm_interest=self.generate_longterm_interest(uids,X_item,X_long,longterm_mask) # bse

        #print(longterm_interest)

        X_user = tf.tile(X_user, [1,tf.shape(shortterm_insterest)[1],1])
        X_combined = tf.concat([X_item, X_user,shortterm_insterest,longterm_interest], axis=2)

        output = self.final_mlp(X_combined)

        result = {'output': output}
        return result


if __name__ == '__main__':
    inputs = {
        'uid': tf.constant([0, 1, 2]),
        'utag1': tf.constant([6, 7, 8]),
        'utag2': tf.constant([9, 10, 11]),
        'utag3': tf.constant([12, 13, 14]),
        'utag4': tf.constant([15, 16, 17]),
        'longterm_goods_ids': tf.constant(
            [[30, 31, 32, 70, 71, 72, 73, 0, 0, 0], [33, 34, 35, 36, 81, 82, 83, 84, 0, 0],
             [37, 38, 39, 40, 41, 91, 92, 93, 94, 0]]),
        'longterm_shop_ids': tf.constant([[42, 43, 44, 70, 71, 72, 73, 0, 0, 0], [45, 46, 47, 48, 81, 82, 83, 84, 0, 0],
                                          [49, 50, 51, 52, 53, 91, 92, 93, 94, 0]]),
        'longterm_cate_ids': tf.constant([[54, 55, 56, 70, 71, 72, 73, 0, 0, 0], [57, 58, 59, 60, 81, 82, 83, 84, 0, 0],
                                          [61, 62, 63, 64, 65, 91, 92, 93, 94, 0]]),
        'shortterm_goods_ids': tf.constant([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0],
                                            [8, 9, 10, 11, 12]]),
        'shortterm_shop_ids': tf.constant([[13, 14, 15, 16, 0], [17, 18, 19, 0, 0],
                                           [20, 21, 22, 23, 24]]),
        'shortterm_cate_ids': tf.constant([[25, 26, 27, 28, 0], [29, 30, 31, 0, 0],
                                           [32, 33, 34, 35, 36]]),
        'label_cate_ids': tf.constant([[11, 12, 13,14,15],[16,17,18,19,20],[21,22,23,24,25]]),
        'label_goods_ids': tf.constant([[31, 32, 33,34,35],[41, 42, 43,44,45],[51, 52, 53,54,55]]),
        'label_shop_ids': tf.constant([[61, 62, 63,64,65],[71, 72, 73,74,75],[81, 82, 83,84,85]]),
    }
    user_and_context_categorical_features = ['uid', 'utag1', 'utag2', 'utag3', 'utag4'],
    item_categorical_features = ['label_goods_ids', 'label_shop_ids', 'label_cate_ids'],
    longterm_series_features = ['longterm_goods_ids', 'longterm_shop_ids', 'longterm_cate_ids'],
    shortterm_series_features = ['shortterm_goods_ids', 'shortterm_shop_ids', 'shortterm_cate_ids']

    layer = SDIMLayer(embedding_dims=4)
    print(layer(inputs))
    print('****************')
    input_dict = {}
    shortterm_max_length = 5
    longterm_max_length = 10
    batch_candidates_num=5
    for feature in layer.user_and_context_categorical_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.shortterm_series_features:
        input_dict[feature] = tf.keras.Input(shape=(shortterm_max_length,), name=feature, dtype=tf.int64)
    for feature in layer.longterm_series_features:
        input_dict[feature] = tf.keras.Input(shape=(longterm_max_length,), name=feature, dtype=tf.int64)
    for feature in layer.item_categorical_features:
        input_dict[feature] = tf.keras.Input(shape=(batch_candidates_num,), name=feature, dtype=tf.int64)
    output = layer(input_dict)
    model = tf.keras.Model(input_dict, output)
    model.summary()
    print(model(inputs))
