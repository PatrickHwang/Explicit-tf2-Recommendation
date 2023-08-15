import tensorflow as tf
from tensorflow.python.ops.math_ops import MatMul
from tensorflow.python.ops.nn_ops import BiasAdd
from tensorflow.python.keras import activations, initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization

from tensorflow.python.framework.errors_impl import InvalidArgumentError as wrapError

import numpy as np

assert tf.__version__[0] == '2', 'model is built by tensorflow2,please check tf version first.'


# tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
class MLPLayer(tf.keras.layers.Layer):
    """
    mlp=MLPLayer([16,4],'tanh')
    input=tf.constant(np.arange(24).reshape(6,4),dtype=float)
    print(mlp(input))
    """

    def __init__(self, units, activation='relu', use_bias=True, is_batch_norm=False,
                 is_dropput=0, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', **kwargs):
        super(MLPLayer, self).__init__(**kwargs)

        self.units = [units] if not isinstance(units, list) else units
        if len(self.units) <= 0:
            raise ValueError(
                f'Received an invalid value for `units`, expected '
                f'a positive integer, got {units}.')
        self.use_bias = use_bias
        self.is_batch_norm = is_batch_norm
        self.is_dropout = is_dropput
        # 调用标准激活函数和初始化方式的常用api
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        # 进入全连接层的输入维度
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')

        dims = [last_dim] + self.units

        self.kernels = []
        self.biases = []
        self.bns = []

        # 1层输入，len(dims)-1层全连接层及对应输出
        for i in range(len(dims) - 1):
            # 第i层的全部参数
            self.kernels.append(
                self.add_weight(f'kernel_{i}',
                                shape=[dims[i], dims[i + 1]],
                                initializer=self.kernel_initializer,
                                trainable=True))

            if self.use_bias:
                self.biases.append(
                    self.add_weight(f'bias_{i}',
                                    shape=[dims[i + 1], ],
                                    initializer=self.bias_initializer,
                                    trainable=True))

            self.bns.append(tf.keras.layers.BatchNormalization())
        self.built = True

    def call(self, inputs, is_train=False):
        _input = inputs
        for i in range(len(self.units)):
            _input = MatMul(a=_input, b=self.kernels[i])
            if self.use_bias:
                _input = BiasAdd(value=_input, bias=self.biases[i])
            if self.is_batch_norm:
                _input = self.bns[i](_input)
            if self.activation is not None:
                _input = self.activation(_input)
            if is_train and self.is_dropout > 0:
                _input = Dropout(self.is_dropout)(_input)  # is_dropout是丢弃率，不是通过率
        return _input


class FMRankingLayer(tf.keras.layers.Layer):
    def __init__(self, feature_names=['item_tag1', 'item_tag2', 'item_tag3'], feature_dims=20, embedding_dims=16,
                 **kwargs):
        super(FMRankingLayer, self).__init__(**kwargs)
        self.feature_names = feature_names
        self.feature_dims = feature_dims
        self.embedding_dims = embedding_dims
        self.bias = self.add_weight(name='bias',
                                    shape=(1,),
                                    trainable=True)
        # embedding and w
        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims,
                                               embeddings_regularizer="l2")
        self.w = tf.keras.layers.Embedding(self.feature_dims,
                                           1,
                                           embeddings_regularizer="l2")

    def build(self, input_shape):
        # bias
        self.bias = self.add_weight(name='bias',
                                    shape=(1,),
                                    trainable=True)
        # embedding and w
        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims,
                                               embeddings_regularizer="l2")
        self.w = tf.keras.layers.Embedding(self.feature_dims,
                                           1,
                                           embeddings_regularizer="l2")
        self.built = True

    def call(self, inputs):
        X = []
        for feature in self.feature_names:
            X.append(inputs[feature])
        X = tf.concat(X, axis=1)

        w_output = self.w(X)
        emb_output = self.embed(X)

        first_order = tf.reduce_sum(w_output, axis=1)

        sum_of_square = tf.reduce_sum(tf.square(emb_output), axis=1)
        square_of_sum = tf.square(tf.reduce_sum(emb_output, axis=1))
        second_order = 0.5 * tf.reduce_sum(tf.subtract(square_of_sum, sum_of_square), axis=1, keepdims=True)

        output = tf.nn.sigmoid(self.bias + first_order + second_order)
        result = {'output': output}
        return result


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


class DinActivationLayer(tf.keras.layers.Layer):
    def __init__(self,activation='PReLU',**kwargs):
        super(DinActivationLayer, self).__init__(**kwargs)
        self.mlp_layer=make_mlp_layer([36],activation=activation, normalization='none')
        self.output_layer=tf.keras.layers.Dense(1)

    def build(self, input_shape):
        super().build(input_shape)


    def call(self,inputs):
        vec1,vec2=inputs
        diff_vec=vec1-vec2
        outer_prod_vec = tf.expand_dims(vec1,1)*tf.expand_dims(vec2,2)  # outer product
        outer_prod_vec_flat = tf.reshape(outer_prod_vec, [tf.shape(outer_prod_vec)[0], tf.shape(outer_prod_vec)[1] * tf.shape(outer_prod_vec)[2]])
        concated_vec=tf.concat([vec1, diff_vec, vec2,  outer_prod_vec_flat], axis=1)
        score=self.output_layer(self.mlp_layer(concated_vec))
        return score


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


class DINLayer(tf.keras.layers.Layer):
    def __init__(self, user_and_context_categorical_features=['uid',  'utag1', 'utag2', 'utag3', 'utag4'],
                 item_categorical_features=['i_goods_id', 'i_shop_id', 'i_cate_id'],
                 behavior_series_features=['visited_goods_ids', 'visited_shop_ids', 'visited_cate_ids'],
                 continuous_features=['itag4_origin', 'itag4_square', 'itag4_cube'],
                 feature_dims=160000, embedding_dims=16,
                 activation='Dice',padding_index=0):
        super(DINLayer, self).__init__()
        self.user_and_context_categorical_features = user_and_context_categorical_features
        self.item_categorical_features = item_categorical_features
        assert len(item_categorical_features) == len(
            behavior_series_features), "Features to be interacted should match in item and behavior series"
        self.behavior_series_features=behavior_series_features
        self.continuous_features=continuous_features
        self.feature_dims = feature_dims
        self.embedding_dims = embedding_dims

        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims)

        self.din_activation_layer = DinActivationLayer(activation=Dice() if activation=='Dice' else activation)

        self.mlp = make_mlp_layer([200, 80],activation=activation,softmax_units=2)

        self.padding_index=padding_index


    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        # X_cont = []
        # for feature in self.continuous_features:
        #     X_cont.append(inputs[feature])
        # X_cont = tf.concat(X_cont, axis=1)

        # 序列无关特征:embedding并展开拼接
        X_cate = []
        for feature in self.user_and_context_categorical_features + self.item_categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_cate.append(feature_tensor)
        X_cate = tf.concat(X_cate, axis=1)
        profile_output = tf.keras.layers.Flatten()(self.embed(X_cate))

        # 序列特征和候选商品特征
        X_item=[]
        for feature in self.item_categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_item.append(feature_tensor)
        X_item = tf.concat(X_item, axis=1)
        X_item = tf.keras.layers.Flatten()(self.embed(X_item))

        # 确定batch中每个样本行为序列的mask
        sequence_mask=inputs[self.behavior_series_features[0]]==self.padding_index

        X_series = tf.stack([inputs[feature] for feature in self.behavior_series_features],
                            axis=2)  # (batch_size, seq_length, feature_cols)
        batch_size, seq_length, feature_cols = X_series.shape
        X_series = tf.reshape(X_series, [-1, feature_cols*seq_length])  # (batch_size,feature_cols*seq_length)
        X_series = self.embed(X_series)   # (batch_size,feature_cols*seq_length,emb_dim)
        X_series=tf.reshape(X_series,[-1,seq_length,feature_cols*self.embedding_dims]) # (batch_size,seq_length,feature_cols*emb_dim)

        # X_series中每一个时刻（第一维)都需要和X_item共同输入到DinActivationLayer,得到seq_length个score
        # 这seq_length个score分别应该和X_series中每一个时刻的embedding做相乘
        # 上一步相乘的结果经过一个sum_pooling的过程，得到一个embedding_dim维的向量X_ineraction

        # 这里使用了tf.vectorized_map，一个可以并行处理序列中每一个时刻的函数
        scores = tf.vectorized_map(lambda x: self.din_activation_layer((X_item, x)), tf.transpose(X_series, perm=[1, 0, 2]))

        # 分数形状是(seq_length, batch_size, 1)，我们需要调整形状以匹配X_series
        scores = tf.transpose(scores, perm=[1, 0, 2])  # now it's (batch_size, seq_length, 1)
        scores = tf.broadcast_to(scores, tf.shape(X_series))  # now it's (batch_size, seq_length, embedding_dim)

        # 应用分数到序列的每个嵌入向量上
        mask = tf.cast(sequence_mask, scores.dtype)
        scores_masked = scores * mask[..., tf.newaxis]
        X_series_scored = X_series * scores_masked

        # 求和得到最终的交互向量
        X_interaction = tf.reduce_sum(X_series_scored, axis=1)


        X_combined = tf.concat([profile_output,X_interaction], axis=1)
        output = self.mlp(X_combined)

        result = {'output': output}
        return result


class DienActivationLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dims1,embedding_dims2=None, **kwargs):
        super(DienActivationLayer, self).__init__(**kwargs)
        if not embedding_dims2:
            embedding_dims2=embedding_dims1
        self.W = self.add_weight(shape=(embedding_dims2, embedding_dims1),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        gru_output, X_item, mask = inputs

        # 计算 gru_output * W * X_item
        weighted_gru_output = tf.linalg.matmul(gru_output, self.W) # 计算gru_output和W的矩阵乘法
        X_item_expanded = tf.expand_dims(X_item, axis=1) # 扩展X_item的维度以进行乘法操作
        product = tf.reduce_sum(weighted_gru_output * X_item_expanded, axis=-1)  # 计算最终的乘积，得到形状为 (batch_size, sequence_length) 的张量

        # 应用 mask 并进行 softmax
        product_exp = tf.exp(product)  # 计算每个元素的指数，这是 softmax 的一部分
        product_exp_masked = product_exp * tf.cast(~mask, dtype=product_exp.dtype)  # 应用 mask，有效步长为1，其他为0
        softmax_denominator = tf.reduce_sum(product_exp_masked, axis=-1, keepdims=True)  # 计算 softmax 的分母
        scores = product_exp_masked / softmax_denominator  # 计算 softmax 结果，即每个步骤的分数

        scores = tf.where(tf.math.is_nan(scores), tf.zeros_like(scores), scores)

        return scores


class CustomGRUCell(tf.keras.layers.Layer):
    def __init__(self, units, mode, **kwargs):
        super(CustomGRUCell, self).__init__(**kwargs)
        self.units = units
        self.mode = mode
        self.state_size = units
        self.gru_cell = tf.keras.layers.GRUCell(units)

    def build(self, input_shape):
        self.W_r = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', name='W_r')
        self.U_r = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', name='U_r')
        self.b_r = self.add_weight(shape=(self.units,), initializer='zeros', name='b_r')

        self.W_h = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', name='W_h')
        self.U_h = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', name='U_h')
        self.b_h = self.add_weight(shape=(self.units,), initializer='zeros', name='b_h')

        if self.mode == "AUGRU":
            self.W_z = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', name='W_z')
            self.U_z = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform', name='U_z')
            self.b_z = self.add_weight(shape=(self.units,), initializer='zeros', name='b_z')

        super(CustomGRUCell, self).build(input_shape)

    def call(self, inputs, states):
        h_prev = states[0]
        x = inputs[:, :self.units]
        a = inputs[:, self.units:]

        r = tf.nn.sigmoid(tf.matmul(x, self.W_r) + tf.matmul(h_prev, self.U_r) + self.b_r)
        h_tilda = tf.nn.tanh(tf.matmul(x, self.W_h) + r * tf.matmul(h_prev, self.U_h) + self.b_h)

        if self.mode == "AGRU":
            h = (1 - a) * h_prev + a * h_tilda
        elif self.mode == "AUGRU":
            u = tf.nn.sigmoid(tf.matmul(x, self.W_z) + tf.matmul(h_prev, self.U_z) + self.b_z)
            u_hat = a * u
            h = (1 - u_hat) * h_prev + u_hat * h_tilda

        return h, [h]


class DienGRU(tf.keras.layers.Layer):
    def __init__(self, units, mode, **kwargs):
        super(DienGRU, self).__init__(**kwargs)
        self.units = units
        self.mode = mode

    def build(self, input_shape):
        self.batch_size, self.seq_len, _ = input_shape[0]
        if self.mode == "AIGRU":
            self.gru = tf.keras.layers.GRU(self.units, return_sequences=True,return_state=True)
        elif self.mode in ["AGRU", "AUGRU"]:
            self.gru_cell = CustomGRUCell(self.units, self.mode)
        super(DienGRU, self).build(input_shape)

    def call(self, inputs, valid_mask):
        gru_input, activation_scores = inputs

        if self.mode == "AIGRU":
            gru_input *= tf.expand_dims(activation_scores, axis=-1)
            _, final_output = self.gru(gru_input, mask=valid_mask)
        else:
            gru_input = tf.concat([gru_input, tf.expand_dims(activation_scores, axis=-1)], axis=-1)
            _, final_output = tf.keras.layers.RNN(self.gru_cell, return_state=True)(gru_input, mask=valid_mask)

        return final_output


class DIENLayer(tf.keras.layers.Layer):
    def __init__(self, user_and_context_categorical_features=['uid',  'utag1', 'utag2', 'utag3', 'utag4'],
                 item_categorical_features=['i_goods_id', 'i_shop_id', 'i_cate_id'],
                 behavior_series_features=['visited_goods_ids', 'visited_shop_ids', 'visited_cate_ids'],
                 negative_sample_features=['negative_goods_ids', 'negative_shop_ids', 'negative_cate_ids'],
                 # continuous_features=['itag4_origin', 'itag4_square', 'itag4_cube'],
                 feature_dims=160000, embedding_dims=16,
                 activation='Dice',padding_index=0,mode="AUGRU",stage='train'):
        super(DIENLayer, self).__init__()
        self.user_and_context_categorical_features = user_and_context_categorical_features
        self.item_categorical_features = item_categorical_features
        assert len(item_categorical_features) == len(
            behavior_series_features)== len(
            negative_sample_features), "Features to be interacted should match in item and behavior/negative series"
        self.behavior_series_features=behavior_series_features
        self.negative_sample_features=negative_sample_features
        #self.continuous_features=continuous_features
        self.feature_dims = feature_dims
        self.embedding_dims = embedding_dims

        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims)

        self.gru = tf.keras.layers.GRU(len(behavior_series_features)*self.embedding_dims, return_sequences=True)

        self.dien_activation_layer = DienActivationLayer(embedding_dims1=len(behavior_series_features)*embedding_dims)

        assert mode in ("AIGRU","AGRU","AUGRU")

        self.dien_gru=DienGRU(len(behavior_series_features)*embedding_dims,mode)

        self.mlp = make_mlp_layer([200, 80],activation=activation,softmax_units=2)

        self.padding_index=padding_index

        self.stage=stage


    def build(self, input_shape):
        super().build(input_shape)

    def set_stage(self,stage):
        assert stage in ('train','inference')
        self.stage=stage

    def get_auxiliary_loss(self, X_series, gru_output, X_negative, valid_mask):
        # 选择需要的时间步
        X_series = X_series[:, :-1]
        gru_output = gru_output[:, 1:]
        X_negative = X_negative[:, 1:]

        # 计算点积
        pos_logits = tf.nn.sigmoid(tf.reduce_sum(X_series * gru_output, axis=-1))
        neg_logits = tf.nn.sigmoid(tf.reduce_sum(X_series * X_negative, axis=-1))

        # 计算二分类交叉熵
        pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pos_logits, labels=tf.ones_like(pos_logits))
        neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=neg_logits, labels=tf.zeros_like(neg_logits))

        # 考虑mask
        mask = valid_mask[:, 1:]  # 考虑到shift操作，有效位数-1
        pos_loss = tf.reduce_sum(pos_loss * tf.cast(mask, dtype=tf.float32))
        neg_loss = tf.reduce_sum(neg_loss * tf.cast(mask, dtype=tf.float32))

        return pos_loss + neg_loss

    def call(self, inputs):
        # X_cont = []
        # for feature in self.continuous_features:
        #     X_cont.append(inputs[feature])
        # X_cont = tf.concat(X_cont, axis=1)

        # 基础特征处理和形状转换
        X_cate = []
        for feature in self.user_and_context_categorical_features + self.item_categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_cate.append(feature_tensor)
        X_cate = tf.concat(X_cate, axis=1)
        profile_output = tf.keras.layers.Flatten()(self.embed(X_cate))

        X_item=[]
        for feature in self.item_categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_item.append(feature_tensor)
        X_item = tf.concat(X_item, axis=1)
        X_item = tf.keras.layers.Flatten()(self.embed(X_item))

        X_series = tf.stack([inputs[feature] for feature in self.behavior_series_features],
                            axis=2)  # (batch_size, seq_length, feature_cols)
        batch_size, seq_length, feature_cols = X_series.shape
        X_series = tf.reshape(X_series, [-1, feature_cols*seq_length])  # (batch_size,feature_cols*seq_length)
        X_series = self.embed(X_series)   # (batch_size,feature_cols*seq_length,emb_dim)
        X_series=tf.reshape(X_series,[-1,seq_length,feature_cols*self.embedding_dims]) # (batch_size,seq_length,feature_cols*emb_dim)

        # 兴趣抽取层——经过标准GRU
        valid_mask = inputs[self.behavior_series_features[0]] != self.padding_index
        gru_output = self.gru(X_series, mask=valid_mask)

        if self.stage=='train':
            X_negative = tf.stack([inputs[feature] for feature in self.negative_sample_features],
                                axis=2)  # (batch_size, seq_length, feature_cols)
            batch_size, seq_length, feature_cols = X_negative.shape
            X_negative = tf.reshape(X_negative, [-1, feature_cols * seq_length])  # (batch_size,feature_cols*seq_length)
            X_negative = self.embed(X_negative)  # (batch_size,feature_cols*seq_length,emb_dim)
            X_negative = tf.reshape(X_negative, [-1, seq_length,
                                             feature_cols * self.embedding_dims])  # (batch_size,seq_length,feature_cols*emb_dim)


            # X_series和gru_output/X_negative错位点击并经过sigmoid,阈值是0-1，和1(正)/0(负)作tf.nn.sigmoid_cross_entropy_with_logits
            auxiliary_loss=self.get_auxiliary_loss(X_series, gru_output, X_negative, valid_mask)

        else:
            auxiliary_loss=0

        # scores=softmax(hWe*valid_mask) (batch_size,sequence_length)
        activation_scores=self.dien_activation_layer((gru_output, X_item, valid_mask))

        evolved_interst=self.dien_gru((gru_output, activation_scores),valid_mask)


        X_combined = tf.concat([profile_output,evolved_interst], axis=1)
        output = self.mlp(X_combined)

        result = {'X_combined':X_combined,'output': output,'auxiliary_loss':auxiliary_loss}
        return result

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, inner_dim, dropout):
        super(EncoderBlock, self).__init__()

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=inner_dim, dropout=dropout
        )
        self.dense = tf.keras.layers.Dense(inner_dim)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, mask=None):
        attn_output = self.attention(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        dense_output = self.dense(out1)
        dense_output = self.dropout2(dense_output)
        return self.layernorm2(out1 + dense_output)


class DSINLayer(tf.keras.layers.Layer):
    def __init__(self, user_and_context_categorical_features=['uid',  'utag1', 'utag2', 'utag3', 'utag4'],
                 item_categorical_features=['i_goods_id', 'i_shop_id', 'i_cate_id'],
                 session_behavior_features=['session_goods_ids', 'session_shop_ids', 'session_cate_ids'],
                 feature_dims=160000, embedding_dims=16,
                 activation='Dice',padding_index=0,dropout_rate=0.1):
        super(DSINLayer, self).__init__()
        self.user_and_context_categorical_features = user_and_context_categorical_features
        self.item_categorical_features = item_categorical_features
        assert len(item_categorical_features) == len(
            session_behavior_features), "Features to be interacted should match in item and behavior series"
        self.session_behavior_features=session_behavior_features
        self.feature_dims = feature_dims
        self.embedding_dims = embedding_dims

        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims)

        self.mlp = make_mlp_layer([200, 80],activation=activation,softmax_units=2)

        self.padding_index=padding_index

        self.dropout_rate=dropout_rate

    def build(self, input_shape):
        super().build(input_shape)
        _,sessions_num,behavior_num=input_shape[self.session_behavior_features[0]]
        embedding_num=self.embedding_dims*len(self.session_behavior_features)

        self.sessions_embedding = tf.keras.layers.Embedding(
            input_dim=sessions_num,
            output_dim=1,
            input_length=1
        )

        self.behavior_embedding = tf.keras.layers.Embedding(
            input_dim=behavior_num,
            output_dim=1,
            input_length=1
        )

        self.feature_cols_embedding = tf.keras.layers.Embedding(
            input_dim=embedding_num,
            output_dim=1,
            input_length=1
        )

        self.encoder_block = EncoderBlock(
            num_heads=8,
            inner_dim=embedding_num,
            dropout=self.dropout_rate
        )

        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_num, return_sequences=True))

        self.dsin_activation_layer1 = DienActivationLayer(
            embedding_dims1=embedding_num)
        self.dsin_activation_layer2 = DienActivationLayer(
            embedding_dims1=embedding_num,embedding_dims2=2 * embedding_num)

        sessions_indices = tf.range(sessions_num)
        sessions_emb = self.sessions_embedding(sessions_indices)  # Shape: (sessions_num, 1)
        self.sessions_emb = tf.reshape(sessions_emb, [1, sessions_num, 1,1])  # Shape: (1, sessions_num, 1)

        # behavior_embedding
        behavior_indices = tf.range(behavior_num)
        behavior_emb = self.behavior_embedding(behavior_indices)  # Shape: (behavior_num, 1)
        self.behavior_emb = tf.reshape(behavior_emb, [1, 1, behavior_num,1])  # Shape: (1, 1, behavior_num, 1)

        # feature_cols_embedding
        feature_cols_indices = tf.range(embedding_num)
        feature_cols_emb = self.feature_cols_embedding(feature_cols_indices)  # Shape: (emb_dim, 1)
        self.feature_cols_emb = tf.reshape(feature_cols_emb, [1, 1, 1, embedding_num])  # Shape: (1, 1, 1, embedding_num)

    def call(self, inputs):

        # 序列无关特征:embedding并展开拼接
        X_cate = []
        for feature in self.user_and_context_categorical_features + self.item_categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_cate.append(feature_tensor)
        X_cate = tf.concat(X_cate, axis=1)
        profile_output = tf.keras.layers.Flatten()(self.embed(X_cate))

        # 序列特征和候选商品特征
        X_item=[]
        for feature in self.item_categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_item.append(feature_tensor)
        X_item = tf.concat(X_item, axis=1)
        X_item = tf.keras.layers.Flatten()(self.embed(X_item))

        # 确定batch中每个样本行为序列的mask
        session_mask = tf.reduce_all(inputs[self.session_behavior_features[0]] == self.padding_index, axis=-1)
        behavior_mask = inputs[self.session_behavior_features[0]] == self.padding_index

        X_series = tf.stack([inputs[feature] for feature in self.session_behavior_features],
                            axis=3)  # (batch_size, sessions_num,behavior_num, feature_cols)
        batch_size, sessions_num,behavior_num, feature_cols = X_series.shape
        X_series = tf.reshape(X_series, [-1, sessions_num*behavior_num*feature_cols])  # (batch_size,feature_cols*seq_length)
        X_series = self.embed(X_series)   # (batch_size,sessions_num*behavior_num*feature_cols,emb_dim)
        X_series=tf.reshape(X_series,[-1,sessions_num,behavior_num,feature_cols*self.embedding_dims]) # (batch_size,sessions_num,behavior_num,feature_cols*emb_dim)

        # 第1，2，3维分别对各种取值定义一个embedding，利用广播机制，embedding可以被学习
        X_series=X_series+self.sessions_emb+self.behavior_emb+self.feature_cols_emb

        # Reshape inputs and mask for the attention layer
        X_series = tf.reshape(X_series, [-1, behavior_num,
                                         feature_cols * self.embedding_dims])  # (batch_size*sessions_num, behavior_num, feature_cols*emb_dim)
        behavior_mask = tf.reshape(behavior_mask, [-1, behavior_num, 1])
        attentioned_X_series=self.encoder_block(X_series,~behavior_mask)
        print(attentioned_X_series)

        # Reshape back to original shape after attention
        attentioned_X_series = tf.reshape(attentioned_X_series,
                                          [-1, sessions_num, behavior_num, feature_cols * self.embedding_dims])

        sessioned_X_series = tf.reduce_mean(attentioned_X_series, axis=2) # (batch_size,session_num,embedding_num)
        bilstmed_X_series = self.bilstm(sessioned_X_series, mask=session_mask)

        sessioned_X_series_scores = self.dsin_activation_layer1([sessioned_X_series, X_item, session_mask])
        bilstmed_X_series_scores = self.dsin_activation_layer2([bilstmed_X_series, X_item, session_mask])

        sessioned_X_series_weighted = tf.reduce_sum(sessioned_X_series * tf.expand_dims(sessioned_X_series_scores, -1),
                                                    axis=1)
        bilstmed_X_series_weighted = tf.reduce_sum(bilstmed_X_series * tf.expand_dims(bilstmed_X_series_scores, -1),
                                                   axis=1)


        X_combined = tf.concat([profile_output,sessioned_X_series_weighted,bilstmed_X_series_weighted], axis=1)
        output = self.mlp(X_combined)

        result = {'output': output}
        return result

class LabelAwareAttentionLayer(tf.keras.layers.Layer):
    def __init__(self,activation='PReLU',**kwargs):
        super(DinActivationLayer, self).__init__(**kwargs)
        self.mlp_layer=make_mlp_layer([36],activation=activation, normalization='none')
        self.output_layer=tf.keras.layers.Dense(1)

    def build(self, input_shape):
        super().build(input_shape)


    def call(self,inputs):
        vec1,vec2=inputs
        diff_vec=vec1-vec2
        outer_prod_vec = tf.expand_dims(vec1,1)*tf.expand_dims(vec2,2)  # outer product
        outer_prod_vec_flat = tf.reshape(outer_prod_vec, [tf.shape(outer_prod_vec)[0], tf.shape(outer_prod_vec)[1] * tf.shape(outer_prod_vec)[2]])
        concated_vec=tf.concat([vec1, diff_vec, vec2,  outer_prod_vec_flat], axis=1)
        score=self.output_layer(self.mlp_layer(concated_vec))
        return score



if __name__ == '__main__':
    inputs = {
        'uid': tf.constant([0, 1, 2]),
        'utag1': tf.constant([6, 7, 8]),
        'utag2': tf.constant([9, 10, 11]),
        'utag3': tf.constant([12, 13, 14]),
        'utag4': tf.constant([15, 16, 17]),
        'itag4_origin': tf.constant([0.2, 7.8, 4.9]),
        'itag4_square': tf.constant([5.3, 1.2, 8.0]),
        'itag4_cube': tf.constant([-3.8, -19.6, 4.2]),
        'i_cate_id':tf.constant([11,12,13]),
        'i_goods_id':tf.constant([14,15,16]),
        'i_shop_id':tf.constant([17,18,19]),
        'visited_goods_ids': tf.constant([[30, 31, 32, 0, 0, 0], [33, 34, 35, 36, 0, 0], [37, 38, 39, 40, 41, 0]]),
        'visited_shop_ids': tf.constant([[42, 43, 44, 0, 0, 0], [45, 46, 47, 48, 0, 0], [49, 50, 51, 52, 53, 0]]),
        'visited_cate_ids': tf.constant([[54, 55, 56, 0, 0, 0], [57, 58, 59, 60, 0, 0], [61, 62, 63, 64, 65, 0]]),
        'negative_goods_ids': tf.constant([[0, 66, 67, 0, 0, 0], [0, 68, 69, 70, 0, 0], [0, 71, 72, 73, 74, 0]]),
        'negative_shop_ids': tf.constant([[0, 75, 76, 0, 0, 0], [0, 77, 78, 79, 0, 0], [0, 80, 81, 82, 83, 0]]),
        'negative_cate_ids': tf.constant([[0, 84, 85, 0, 0, 0], [0, 86, 87, 88, 0, 0], [0, 89, 90, 91, 92, 0]])
    }
    layer=DIENLayer(embedding_dims=5,mode="AIGRU")
    print(layer(inputs))
    #layer = DINLayer(embedding_dims=5)
    # 输入
    max_length=6
    input_dic = {}
    for feature in layer.user_and_context_categorical_features+layer.item_categorical_features:
        input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.continuous_features:
        input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
    for feature in layer.behavior_series_features+layer.negative_sample_features: # DIN: for feature in layer.behavior_series_features:
        input_dic[feature] = tf.keras.Input(shape=(max_length,), name=feature, dtype=tf.int64)


    output = layer(input_dic)
    model = tf.keras.Model(input_dic, output)
    model.summary()
    print(model(inputs))
    """
    gru_output=tf.random.uniform((3,4,5),0,10)
    X_item=tf.random.uniform((3, 5), 0.0, 1.0)
    valid_mask=tf.random.uniform((3,4),0,2, dtype=tf.int32)
    layer=DienActivationLayer(5)
    print(layer((gru_output, X_item, valid_mask)))
    """
    """
    batch_size = 10
    seq_len = 5
    units = 6

    # 随机输入
    valid_mask = tf.constant(np.random.randint(2, size=(batch_size, seq_len)), dtype=tf.float32)
    activation_scores = tf.constant(np.random.randn(batch_size, seq_len), dtype=tf.float32)

    print("Testing CustomGRUCell:")
    for mode in ["AGRU", "AUGRU"]:
        print(f"Mode: {mode}")
        custom_gru_cell = CustomGRUCell(units, mode)
        state = [tf.zeros((batch_size, units))]
        inputs = tf.constant(np.random.randn(batch_size, seq_len, units + 1), dtype=tf.float32)
        for t in range(seq_len):
            output, state = custom_gru_cell(inputs[:, t, :], state)
        print("Final output:", output)

    # 测试 DienGRU
    print("\nTesting DienGRU:")
    for mode in ["AIGRU", "AGRU", "AUGRU"]:
        inputs = tf.constant(np.random.randn(batch_size, seq_len, units), dtype=tf.float32)
        print(f"Mode: {mode}")
        dien_gru = DienGRU(units, mode)
        final_output = dien_gru((inputs, activation_scores), valid_mask)
        print("Final output:", final_output)
    """
    """
    inputs = {'uid': tf.constant([40, 1, 2]),
        'utag1': tf.constant([6, 7, 8]),
        'utag2': tf.constant([9, 10, 11]),
        'utag3': tf.constant([12, 13, 14]),
        'utag4': tf.constant([15, 16, 17]),
        'itag4_origin': tf.constant([0.2, 7.8, 4.9]),
        'itag4_square': tf.constant([5.3, 1.2, 8.0]),
        'itag4_cube': tf.constant([-3.8, -19.6, 4.2]),
        'i_cate_id': tf.constant([11, 12, 13]),
        'i_goods_id': tf.constant([14, 15, 16]),
        'i_shop_id': tf.constant([17, 18, 19]),
        'session_cate_ids': tf.constant([
        [[27, 28, 29, 0, 0], [32, 33, 34, 35, 36], [37, 38, 39, 0, 0]],  # batch 1
        [[40, 41, 0, 0, 0], [45, 46, 47, 0, 0], [0, 0, 0, 0, 0]],  # batch 2
        [[48, 49, 50, 51, 52], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],  # batch 3
    ]),
    'session_goods_ids':tf.constant([
        [[63, 64, 65, 0, 0], [68, 69, 70, 71, 72], [73, 74, 75, 0, 0]],  # batch 1
        [[76, 77, 0, 0, 0], [81, 82, 83, 0, 0], [0, 0, 0, 0, 0]],  # batch 2
        [[84, 85, 86, 87, 88], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],  # batch 3
    ]),
    'session_shop_ids':tf.constant([
        [[99, 100, 101, 0, 0], [104, 105, 106, 107, 108], [109, 110, 111, 0, 0]],  # batch 1
        [[112, 113, 0, 0, 0], [117, 118, 119, 0, 0], [0, 0, 0, 0, 0]],  # batch 2
        [[120, 121, 122, 123, 124], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],  # batch 3
    ])
              }
    input_dic = {}
    layer = DSINLayer(embedding_dims=8)
    max_sessions,max_behavior = 3,5
    for feature in layer.user_and_context_categorical_features + layer.item_categorical_features:
        input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.session_behavior_features:
        input_dic[feature] = tf.keras.Input(shape=(max_sessions, max_behavior,), name=feature, dtype=tf.int64)
    output = layer(input_dic)
    model = tf.keras.Model(input_dic, output)
    model.summary()
    print(model(inputs))
    """
