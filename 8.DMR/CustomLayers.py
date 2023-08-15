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


class DMRLayer(tf.keras.layers.Layer):
    """
    inputs = {
        'uid': tf.constant([0, 1, 2]),
        'utag1': tf.constant([6, 7, 8]),
        'utag2': tf.constant([9, 10, 11]),
        'utag3': tf.constant([12, 13, 14]),
        'utag4': tf.constant([15, 16, 17]),
        'i_cate_id': tf.constant([11, 12, 13]),
        'i_goods_id': tf.constant([14, 15, 16]),
        'i_shop_id': tf.constant([17, 18, 19]),
        'visited_goods_ids': tf.constant([[30, 31, 32,90,91,92,93, 0, 0, 0], [33, 34, 35, 36,80,81,82,83, 0, 0], [37, 38, 39, 40, 41, 70,71,72,73,0]]),
        'visited_shop_ids': tf.constant([[42, 43, 44,60,61,62,63, 0, 0, 0], [45, 46, 47, 48,50,51,52,53, 0, 0], [49, 50, 51, 52, 53,30,31,32,33, 0]]),
        'visited_cate_ids': tf.constant([[54, 55, 56,101,102,103,104, 0, 0, 0], [57, 58, 59, 60,105,106,107,108, 0, 0], [61, 62, 63, 64, 65,111,112,113,114, 0]]),
        'negative_goods_ids': tf.constant([[1, 66, 67, 2, 3, 4], [5, 68, 69, 70, 6, 7], [8, 71, 72, 73, 74, 9]]),
        'negative_shop_ids': tf.constant([[10, 75, 76, 11, 12, 13], [14, 77, 78, 79, 15, 16], [17, 80, 81, 82, 83, 18]]),
        'negative_cate_ids': tf.constant([[19, 84, 85, 20, 21, 22], [23, 86, 87, 88, 24, 25], [26, 89, 90, 91, 92, 27]])
    }
    layer = DMRLayer()
    set_custom_initialization(layer)
    print(layer(inputs))
    print('****************')
    input_dict = {}
    max_length = 10
    neg_samples=6
    for feature in layer.user_and_context_categorical_features + layer.item_categorical_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.behavior_series_features:
        input_dict[feature] = tf.keras.Input(shape=(max_length,), name=feature, dtype=tf.int64)
    for feature in layer.negative_sample_features:
        input_dict[feature] = tf.keras.Input(shape=(neg_samples,), name=feature, dtype=tf.int64)
    output = layer(input_dict)
    model = tf.keras.Model(input_dict, output)
    model.summary()
    print(model(inputs))
    """
    def __init__(self, user_and_context_categorical_features=['uid',  'utag1', 'utag2', 'utag3', 'utag4'],
                 item_categorical_features=['i_goods_id', 'i_shop_id', 'i_cate_id'],
                 behavior_series_features=['visited_goods_ids', 'visited_shop_ids', 'visited_cate_ids'],
                 negative_sample_features=['negative_goods_ids', 'negative_shop_ids', 'negative_cate_ids'],
                 feature_dims=160000, embedding_dims=16,
                 activation='PReLU',padding_index=0,stage='train',max_length=10):
        super(DMRLayer, self).__init__()
        self.user_and_context_categorical_features = user_and_context_categorical_features
        self.item_categorical_features = item_categorical_features
        assert len(item_categorical_features) == len(
            behavior_series_features)== len(
            negative_sample_features), "Features to be interacted should match in item and behavior/negative series"
        self.behavior_series_features=behavior_series_features
        self.negative_sample_features=negative_sample_features
        self.feature_dims = feature_dims
        self.embedding_dims = embedding_dims

        self.id_embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims)

        self.max_length=max_length
        self.pos_embed = tf.keras.layers.Embedding(max_length,len(item_categorical_features)*self.embedding_dims)


        self.mlp = make_mlp_layer([200, 80],activation=activation,sigmoid_units=True)

        self.padding_index=padding_index

        self.stage=stage

        self.i2i_network=DMRI2INetworkLayer(len(item_categorical_features)*self.embedding_dims)
        self.u2i_network=DMRU2INetworkLayer(len(item_categorical_features)*self.embedding_dims)


    def build(self, input_shape):
        super().build(input_shape)

    def set_stage(self,stage):
        assert stage in ('train','valid','inference')
        self.stage=stage


    def call(self, inputs):
        # 基础特征处理和形状转换
        X_cate = []
        for feature in self.user_and_context_categorical_features + self.item_categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_cate.append(feature_tensor)
        X_cate = tf.concat(X_cate, axis=1)
        profile_output = tf.keras.layers.Flatten()(self.id_embed(X_cate))

        X_item=[]
        for feature in self.item_categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_item.append(feature_tensor)
        X_item = tf.concat(X_item, axis=1)
        X_item = tf.keras.layers.Flatten()(self.id_embed(X_item))

        X_series = tf.stack([inputs[feature] for feature in self.behavior_series_features],
                            axis=2)  # (batch_size, seq_length, feature_cols)
        batch_size, seq_length, feature_cols = X_series.shape
        X_series = tf.reshape(X_series, [-1, feature_cols*seq_length])  # (batch_size,feature_cols*seq_length)
        X_series = self.id_embed(X_series)   # (batch_size,feature_cols*seq_length,emb_dim)
        X_series=tf.reshape(X_series,[-1,seq_length,feature_cols*self.embedding_dims]) # (batch_size,seq_length,feature_cols*emb_dim)

        pos_series=self.pos_embed(tf.range(self.max_length)) # (seq_length,feature_cols*emb_dim)

        valid_mask = inputs[self.behavior_series_features[0]] != self.padding_index

        i2i_vector=self.i2i_network([X_series,pos_series,X_item,valid_mask])

        X_negative = tf.stack([inputs[feature] for feature in self.negative_sample_features],
                              axis=2)  # (batch_size, seq_length, feature_cols)
        batch_size, seq_length, feature_cols = X_negative.shape
        X_negative = tf.reshape(X_negative, [-1, feature_cols * seq_length])  # (batch_size,feature_cols*seq_length)
        X_negative = self.id_embed(X_negative)  # (batch_size,feature_cols*seq_length,emb_dim)
        X_negative = tf.reshape(X_negative, [-1, seq_length,
                                             feature_cols * self.embedding_dims])  # (batch_size,seq_length,feature_cols*emb_dim)
        u2i_vector,auxiliary_loss=self.u2i_network([X_series,pos_series,X_item,valid_mask,self.stage,X_negative])

        X_combined = tf.concat([profile_output,i2i_vector,u2i_vector], axis=1)
        output = self.mlp(X_combined)

        result = {'output': output,'auxiliary_loss':auxiliary_loss}
        return result


class DMRI2INetworkLayer(tf.keras.layers.Layer):
    def __init__(self,hidden_dim=48):
        super(DMRI2INetworkLayer, self).__init__()
        self.hidden_dim=hidden_dim

    def build(self, input_shape):
        input_dim=input_shape[0][-1]
        self.Wc = tf.Variable(
            tf.initializers.GlorotUniform()(shape=(input_dim, self.hidden_dim)),
            trainable=True)
        self.Wp = tf.Variable(
            tf.initializers.GlorotUniform()(shape=(input_dim, self.hidden_dim)),
            trainable=True)
        self.We = tf.Variable(
            tf.initializers.GlorotUniform()(shape=(input_dim, self.hidden_dim)),
            trainable=True)
        self.b=tf.Variable(
            tf.initializers.GlorotUniform()(shape=(self.hidden_dim,)),
            trainable=True)
        self.z=tf.Variable(
            tf.initializers.GlorotUniform()(shape=(self.hidden_dim,)),
            trainable=True)

    def call(self, inputs):
        X_series,pos_series,X_item,valid_mask=inputs
        shape = tf.shape(X_series)
        batch_size = shape[0]
        seq_length = shape[1]
        embedding_dim = shape[2]
        We = tf.tile(tf.expand_dims(self.We, 0), [batch_size, 1, 1])
        # 遵循了论文中attention_scores的公式定义
        attention_scores=tf.tile(tf.expand_dims(tf.expand_dims(self.z,axis=0),axis=0),[batch_size,seq_length,1])*\
                         tf.nn.tanh(tf.tile(tf.expand_dims(tf.matmul(X_item,self.Wc),axis=1),[1,seq_length,1])+\
                                           tf.matmul(X_series,We)+\
                                           tf.tile(tf.expand_dims(tf.matmul(pos_series,self.Wp),axis=0),[batch_size,1,1]))
        attention_scores=tf.reduce_sum(attention_scores,axis=-1)   # (batch_size,seq_length)
        # 论文中需要返回有效步的attention之和
        masked_scores = tf.where(valid_mask, attention_scores, tf.zeros_like(attention_scores))
        score_sum = tf.reduce_sum(masked_scores, axis=-1, keepdims=True)  # (batch_size,1)
        # 把有效步的scores做softmax，注意无效步已经加上了极大的负数以免softmax中分到权重
        attention_scores=tf.where(valid_mask,attention_scores,tf.ones_like(attention_scores)*(-np.inf))
        attention_scores=tf.nn.softmax(attention_scores,axis=-1)
        attentioned_vectors=tf.tile(tf.expand_dims(attention_scores,axis=2),[1,1,embedding_dim])*X_series
        attention_output=tf.reduce_sum(attentioned_vectors,axis=1)      # (batch_size,embedding_dim)
        concated_output=tf.concat([attention_output,score_sum],axis=1)  # (batch_size,embedding_dim+1)
        return concated_output


class DMRU2INetworkLayer(tf.keras.layers.Layer):
    def __init__(self,hidden_dim=48):
        super(DMRU2INetworkLayer, self).__init__()
        self.hidden_dim=hidden_dim

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.Wp = tf.Variable(
            tf.initializers.GlorotUniform()(shape=(input_dim, self.hidden_dim)),
            trainable=True)
        self.We = tf.Variable(
            tf.initializers.GlorotUniform()(shape=(input_dim, self.hidden_dim)),
            trainable=True)
        self.b = tf.Variable(
            tf.initializers.GlorotUniform()(shape=(self.hidden_dim,)),
            trainable=True)
        self.z = tf.Variable(
            tf.initializers.GlorotUniform()(shape=(self.hidden_dim,)),
            trainable=True)

    def compute_auxiliary_loss(self,attentioned_vectors,valid_mask,X_negative,neg_mask=None):
        # 得到最后一个有效时间步的输出(正样本)和前T-1步的加权向量表示(用于和正负样本作内积)
        last_true_indices = tf.math.reduce_sum(tf.cast(valid_mask, tf.int32), axis=1) - 1
        batch_indices = tf.range(tf.shape(attentioned_vectors)[0])
        indices = tf.stack([batch_indices, last_true_indices], axis=1)
        last_true_vector = tf.gather_nd(attentioned_vectors, indices)
        new_mask = tf.math.less(tf.range(valid_mask.shape[1]), tf.expand_dims(last_true_indices, axis=-1))  # (batch_size,seq_length)
        output_vector=attentioned_vectors*tf.tile(tf.expand_dims(tf.cast(new_mask,dtype=tf.float32),axis=-1),[1,1,tf.shape(attentioned_vectors)[-1]])
        output_vector=tf.reduce_sum(output_vector,axis=1)  # (batch_size,seq_length)

        pos_logits=tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE,axis=-1)(last_true_vector,output_vector)
        pos_logits=(pos_logits+1)/2
        neg_logits=tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE,axis=-1)(X_negative,tf.tile(tf.expand_dims(output_vector,axis=1),[1,tf.shape(X_negative)[1],1]))
        neg_logits=(neg_logits+1)/2
        pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pos_logits, labels=tf.ones_like(pos_logits))
        neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=neg_logits, labels=tf.zeros_like(neg_logits))
        if not neg_mask:
            neg_mask=tf.ones_like(neg_loss)
        auxiliary_loss=tf.reduce_sum(pos_loss,axis=-1)+tf.reduce_sum(neg_loss*neg_mask,axis=-1)
        return auxiliary_loss

    def call(self, inputs):
        X_series,pos_series,X_item,valid_mask,stage,origin=inputs
        shape = tf.shape(X_series)
        batch_size = shape[0]
        seq_length = shape[1]
        embedding_dim = shape[2]
        We = tf.tile(tf.expand_dims(self.We, 0), [batch_size, 1, 1])
        # 遵循了论文中attention_scores的公式定义
        attention_scores=tf.tile(tf.expand_dims(tf.expand_dims(self.z,axis=0),axis=0),[batch_size,seq_length,1])*\
                         tf.nn.tanh(tf.matmul(X_series,We)+\
                                           tf.tile(tf.expand_dims(tf.matmul(pos_series,self.Wp),axis=0),[batch_size,1,1]))
        attention_scores=tf.reduce_sum(attention_scores,axis=-1)   # (batch_size,seq_length)
        # 把有效步的scores做softmax，注意无效步已经加上了极大的负数以免softmax中分到权重
        attention_scores=tf.where(valid_mask,attention_scores,tf.ones_like(attention_scores)*(-np.inf))
        attention_scores=tf.nn.softmax(attention_scores,axis=-1)
        attentioned_vectors=tf.tile(tf.expand_dims(attention_scores,axis=2),[1,1,embedding_dim])*X_series
        attention_output=tf.reduce_sum(attentioned_vectors,axis=1)      # (batch_size,embedding_dim)
        inner_product=tf.reduce_sum(attention_output*X_item,axis=-1,keepdims=True)
        # 其实原论文是attention_output过ReLU再和X_item作内积，然后只返回内积。这里取消ReLU，并把内积和表示向量作拼接再返回
        concated_output = tf.concat([attention_output, inner_product], axis=1)  # (batch_size,embedding_dim+1)
        if stage=='train' or 'eval':
            auxiliary_loss=self.compute_auxiliary_loss(attentioned_vectors,valid_mask,origin)
        else:
            auxiliary_loss=0
        return concated_output,auxiliary_loss


class FinalMLPLayer(tf.keras.layers.Layer):
    """
    inputs = {'uid': tf.constant([0, 1, 2]), 'iid': tf.constant([3, 4, 5]),
                 'utag1': tf.constant([6, 7, 8]), 'utag2': tf.constant([9, 10, 11]),
                 'utag3': tf.constant([12, 13, 14]), 'utag4': tf.constant([15, 16, 17]),
                 'itag1': tf.constant([18, 19, 20]), 'itag2': tf.constant([21, 22, 23]),
                 'itag3': tf.constant([24, 25, 26]), 'itag4': tf.constant([27, 28, 29])}
    layer = FinalMLPLayer()
    set_custom_initialization(layer)
    print(layer(inputs))
    print('****************')
    input_dict = {}
    for feature in layer.user_features + layer.item_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    output = layer(input_dict)
    model = tf.keras.Model(input_dict, output)
    model.summary()
    print(model(inputs))
    """
    def __init__(self,user_features=['uid', 'utag1', 'utag2', 'utag3', 'utag4'],
                 item_features=['iid','itag1', 'itag2', 'itag3','itag4'],feature_dims=160000,embedding_dims=16,
                 fs1_feature_dims=160000,fs2_feature_dims=160000,fs1_emd_dim=16,fs2_emd_dim=16,fs1_hidden_units=[64],fs2_hidden_units=[64],
                 fs1_activation='ReLU',fs2_activation='ReLU',fs1_norm='None',fs2_norm='None',
                 mlp1_hidden_units=[64,64,64],mlp2_hidden_units=[64,64,64],mlp1_activation='ReLU',mlp2_activation='ReLU',
                 mlp1_norm='batchnorm',mlp2_norm='batchnorm',num_heads=8,output_dim=1):
        super(FinalMLPLayer, self).__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.shared_embedding = tf.keras.layers.Embedding(feature_dims,embedding_dims)
        fs1_units=fs1_hidden_units+[len(user_features+item_features)*embedding_dims]
        fs2_units = fs2_hidden_units + [len(user_features + item_features) * embedding_dims]
        self.fs_layer=FeatureSelectionLayer(fs1_feature_dims=fs1_feature_dims,fs2_feature_dims=fs2_feature_dims,
                                            fs1_emd_dim=fs1_emd_dim,fs2_emd_dim=fs2_emd_dim,
                                            fs1_units=fs1_units,fs2_units=fs2_units,
                 fs1_activation=fs1_activation,fs2_activation=fs2_activation,fs1_norm=fs1_norm,fs2_norm=fs2_norm)
        self.mlp1=make_mlp_layer(mlp1_hidden_units, mlp1_activation, mlp1_norm)
        self.mlp2=make_mlp_layer(mlp2_hidden_units, mlp2_activation, mlp2_norm)

        self.interaction_layer=DualPartsInteractionLayer(num_heads,output_dim)

    def call(self, inputs):
        X_user = []
        for feature in self.user_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_user.append(feature_tensor)
        X_user = tf.concat(X_user, axis=1)

        X_item = []
        for feature in self.item_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_item.append(feature_tensor)
        X_item = tf.concat(X_item, axis=1)

        X_shared = tf.concat([X_user,X_item],axis=1)
        X_shared = tf.keras.layers.Flatten()(self.shared_embedding(X_shared))
        X_part1,X_part2=self.fs_layer([X_shared,X_item,X_user])

        mlp_part1=self.mlp1(X_part1)
        mlp_part2=self.mlp2(X_part2)

        output=self.interaction_layer((mlp_part1,mlp_part2))
        output=tf.nn.sigmoid(output)

        return {'output':output}


class FeatureSelectionLayer(tf.keras.layers.Layer):
    def __init__(self, fs1_feature_dims=160000, fs2_feature_dims=160000, fs1_emd_dim=16, fs2_emd_dim=16,
                 fs1_units=None,fs2_units=None,fs1_activation='ReLU', fs2_activation='ReLU',
                 fs1_norm='None', fs2_norm='None'):
        super(FeatureSelectionLayer, self).__init__()
        if fs2_units is None:
            fs2_units = [64, 160]
        if fs1_units is None:
            fs1_units = [64, 160]
        self.fs1_embedding = tf.keras.layers.Embedding(fs1_feature_dims, fs1_emd_dim)
        self.fs2_embedding = tf.keras.layers.Embedding(fs2_feature_dims, fs2_emd_dim)
        self.mlp_gate1 = make_mlp_layer(fs1_units, fs1_activation, fs1_norm)
        self.mlp_gate2 = make_mlp_layer(fs2_units, fs2_activation, fs2_norm)

    def call(self, inputs):
        X_shared,X_item,X_user=inputs
        X_item=self.mlp_gate1(tf.keras.layers.Flatten()(self.fs1_embedding(X_item)))
        X_user=self.mlp_gate2(tf.keras.layers.Flatten()(self.fs2_embedding(X_user)))
        X_item=tf.nn.sigmoid(X_item)*2
        X_user = tf.nn.sigmoid(X_user) * 2
        out1=X_item*X_shared
        out2=X_user*X_shared
        return out1,out2

class DualPartsInteractionLayer(tf.keras.layers.Layer):
    def __init__(self,num_heads=8,output_dim=1):
        super(DualPartsInteractionLayer, self).__init__()
        self.num_heads=num_heads
        self.output_dim=output_dim
        self.W_1=tf.keras.layers.Dense(output_dim)
        self.W_2 = tf.keras.layers.Dense(output_dim)

    def build(self,input_shape):
        dim1=input_shape[0][-1]
        dim2=input_shape[1][-1]
        assert dim1%self.num_heads==dim2%self.num_heads==0,"输入维度必须被num_heads整除"
        self.head_dim1=dim1//self.num_heads
        self.head_dim2=dim2//self.num_heads
        self.W_12=tf.Variable(tf.initializers.GlorotUniform()(shape=(self.num_heads, self.head_dim1, self.head_dim2,self.output_dim)), trainable=True)

    def call(self,inputs):
        mlp_part1,mlp_part2=inputs
        output_1=self.W_1(mlp_part1)
        output_2=self.W_2(mlp_part2)
        mlp_part1=tf.reshape(mlp_part1,(-1,self.num_heads,self.head_dim1))
        mlp_part2=tf.reshape(mlp_part2,(-1,self.num_heads,self.head_dim2))
        output_12=tf.einsum('bnx,nxyo->bnyo',mlp_part1,self.W_12)
        output_12=tf.einsum('bnyo,bny->bno',output_12,mlp_part2)
        output_12=tf.reduce_sum(output_12,axis=1)
        output=output_1+output_2+output_12
        return output


class DMTLayer(tf.keras.layers.Layer):
    def __init__(self, user_and_context_categorical_features=['uid',  'utag1', 'utag2', 'utag3', 'utag4'],
                 item_categorical_features=['i_goods_id', 'i_shop_id', 'i_cate_id'],
                 continuous_features=['continuous_feature1','continuous_feature2','continuous_feature3'],
                 click_series_features=['click_goods_ids', 'click_shop_ids', 'click_cate_ids'],
                 cart_series_features=['cart_goods_ids', 'cart_shop_ids', 'cart_cate_ids'],
                 order_series_features=['order_goods_ids', 'order_shop_ids', 'order_cate_ids'],
                 position_bias_features={'position_feature':'position','page_feature':'page','neighbourhood_features':['near_expo_seq_cate2','near_expo_seq_cate3']},
                 feature_dims=160000, embedding_dims=16,
                 training=True,tf_dim=48,tf_layers_num=3,tf_heads_num=4,tf_dff=32,
                 expert_units=[96,32],expert_activation='PReLU',expert_num=6,
                 gate_units=[64],gate_activation='PReLU',output_units=[8,1],output_activation='PReLU',dropout_rate=0.1,
                 max_position=400,max_page=100,neighbour_num=7):
        super(DMTLayer, self).__init__()
        self.user_and_context_categorical_features = user_and_context_categorical_features
        self.item_categorical_features = item_categorical_features
        assert len(item_categorical_features) == len(
            click_series_features)==len(cart_series_features)==len(order_series_features), "Features to be interacted should match in item and behavior series"
        self.continuous_features=continuous_features
        self.click_series_features=click_series_features
        self.cart_series_features=cart_series_features
        self.order_series_features=order_series_features
        self.position_bias_features=position_bias_features
        self.feature_dims = feature_dims
        self.embedding_dims = embedding_dims

        self.id_embed = Embedding(self.feature_dims,self.embedding_dims)

        self.click_transformer = DeepInterestTransformer(num_layers=tf_layers_num, d_model=tf_dim, num_heads=tf_heads_num,
                                             dff=tf_dff,shared_embedding_layer=self.id_embed)
        self.cart_transformer = DeepInterestTransformer(num_layers=tf_layers_num, d_model=tf_dim, num_heads=tf_heads_num,
                                             dff=tf_dff, shared_embedding_layer=self.id_embed)
        self.order_transformer = DeepInterestTransformer(num_layers=tf_layers_num, d_model=tf_dim, num_heads=tf_heads_num,
                                             dff=tf_dff, shared_embedding_layer=self.id_embed)

        self.expert_num=expert_num
        self.expert_model = [make_mlp_layer(expert_units,expert_activation,dropout_rate=dropout_rate) for _ in range(expert_num)]
        self.ctr_gate = make_mlp_layer(gate_units,gate_activation,softmax_units=expert_num,dropout_rate=dropout_rate)
        self.cvr_gate = make_mlp_layer(gate_units, gate_activation, softmax_units=expert_num,dropout_rate=dropout_rate)

        self.ctr_output = make_mlp_layer(output_units,output_activation,dropout_rate=dropout_rate)
        self.cvr_output = make_mlp_layer(output_units,output_activation,dropout_rate=dropout_rate)

        self.bias_network_ctr=BiasDeepNeuralNetworkLayer(position_bias_features=position_bias_features,
                                                     feature_dims=feature_dims, embedding_dims=embedding_dims,
                                                     max_position=max_position,max_page=max_page,neighbour_num=neighbour_num)
        self.bias_network_cvr = BiasDeepNeuralNetworkLayer(position_bias_features=position_bias_features,
                                                           feature_dims=feature_dims, embedding_dims=embedding_dims,
                                                           max_position=max_position, max_page=max_page,
                                                           neighbour_num=neighbour_num)

        self.training=training

    def build(self, input_shape):
        super().build(input_shape)

    def set_stage(self,training):
        self.training=training

    def process_basic_features(self,inputs,features,embedding=True):
        X=[]
        for feature in features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X.append(feature_tensor)
        X = tf.concat(X, axis=1)
        if embedding:
            X = tf.keras.layers.Flatten()(self.id_embed(X))
        return X

    def process_series_features(self,inputs,features):
        X_series = tf.stack([inputs[feature] for feature in features],
                            axis=2)  # (batch_size, seq_length, feature_cols)
        batch_size, seq_length, feature_cols = X_series.shape
        X_series = tf.reshape(X_series, [-1, feature_cols * seq_length])  # (batch_size,feature_cols*seq_length)
        X_series = self.id_embed(X_series)  # (batch_size,feature_cols*seq_length,emb_dim)
        X_series = tf.reshape(X_series, [-1, seq_length,
                                         feature_cols * self.embedding_dims])  # (batch_size,seq_length,feature_cols*emb_dim)
        return X_series


    def call(self, inputs):
        # 组装特征部分
        X_cate=self.process_basic_features(inputs,self.user_and_context_categorical_features + self.item_categorical_features,True)
        X_cont=self.process_basic_features(inputs,self.continuous_features,False)
        X_item = self.process_basic_features(inputs,self.item_categorical_features,True)

        X_click=self.process_series_features(inputs,self.click_series_features)
        X_cart=self.process_series_features(inputs,self.cart_series_features)
        X_order=self.process_series_features(inputs,self.order_series_features)

        # DeepInterest Transformer部分
        X_interest=[]
        for transformer,features,embed_features in zip([self.click_transformer,self.cart_transformer,self.order_transformer],\
                [self.click_series_features,self.cart_series_features,self.order_series_features],\
                                                       [X_click,X_cart,X_order]):
            encoder_input=inputs[features[0]]
            decoder_input=tf.expand_dims(inputs[self.item_categorical_features[0]],axis=1)
            encoder_input_embedded = embed_features
            decoder_input_embedded = tf.expand_dims(X_item,axis=1)
            interest=transformer(encoder_input,decoder_input,self.training,encoder_input_embedded,decoder_input_embedded)
            interest=tf.squeeze(interest,axis=1)
            X_interest.append(interest)
        X_concated=X_interest+[X_cate,X_cont,X_item]
        X_concated = tf.concat(X_concated, axis=1)

        # MMOE部分
        expert_outputs = []
        for i in range(self.expert_num):
            expert_output = self.expert_model[i](X_concated)
            expert_outputs.append(expert_output)

        ctr_gate = self.ctr_gate(X_concated)
        cvr_gate = self.cvr_gate(X_concated)

        expert_outputs = tf.stack(expert_outputs, axis=1)  # (batch,expert_num,expert_dim)
        ctr_gate = tf.expand_dims(ctr_gate, axis=2)  # (batch,expert_num,1)
        cvr_gate = tf.expand_dims(cvr_gate, axis=2)  # (batch,expert_num,1)
        ctr_input = tf.keras.layers.Flatten()(expert_outputs * ctr_gate)
        cvr_input = tf.keras.layers.Flatten()(expert_outputs * cvr_gate)

        ctr_logits=self.ctr_output(ctr_input)
        cvr_logits=self.cvr_output(cvr_input)

        ctr_bias=self.bias_network_ctr(inputs)
        cvr_bias=self.bias_network_cvr(inputs)

        ctr_unbias=tf.nn.sigmoid(ctr_logits+ctr_bias)
        cvr_unbias=tf.nn.sigmoid(cvr_logits+cvr_bias)

        result = {'ctr_logits': ctr_unbias,'cvr_logits': cvr_unbias}
        return result


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(num_heads, d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, attention_mask=mask)  # 修改此行
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

    def call(self, x, training, mask):
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(num_heads, d_model)
        self.mha2 = MultiHeadAttention(num_heads, d_model)

        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

        return x  # (batch_size, target_seq_len, d_model)


class DeepInterestTransformer(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size=None,
                 rate=0.1, shared_embedding_layer=None):
        super(DeepInterestTransformer, self).__init__()

        if shared_embedding_layer:
            self.embedding = shared_embedding_layer
        else:
            self.embedding = Embedding(input_dim=input_vocab_size, output_dim=d_model)

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, rate)

    def call(self, encoder_input, decoder_input, training,encoder_input_embedded=None,decoder_input_embedded=None):
        if encoder_input_embedded is None:
            encoder_input_embedded = self.embedding(encoder_input)
        if decoder_input_embedded is None:
            decoder_input_embedded = self.embedding(decoder_input)

        encoder_padding_mask = self.create_encoder_padding_mask(encoder_input)
        decoder_look_ahead_mask = self.create_decoder_mask(decoder_input)
        decoder_padding_mask = self.create_decoder_padding_mask(encoder_input, decoder_input)

        encoder_output = self.encoder(encoder_input_embedded, training, encoder_padding_mask)
        decoder_output = self.decoder(decoder_input_embedded, encoder_output, training, decoder_look_ahead_mask, decoder_padding_mask)

        return decoder_output

    def create_encoder_padding_mask(self, seq):
        mask = tf.cast(~tf.math.equal(seq, 0), tf.float32)
        mask_expanded = mask[:, tf.newaxis, :] * mask[:, :, tf.newaxis]
        return mask_expanded

    def create_padding_mask(self, seq):
        mask = tf.cast(~tf.math.equal(seq, 0), tf.float32)
        return mask

    def create_decoder_look_ahead_mask(self, size):
        mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def create_decoder_mask(self, seq):
        seq_len = tf.shape(seq)[1]
        look_ahead_mask = self.create_decoder_look_ahead_mask(seq_len)
        padding_mask = self.create_padding_mask(seq)
        return look_ahead_mask * padding_mask[:,:,tf.newaxis]

    def create_decoder_padding_mask(self, encoder_seq, decoder_seq):
        if len(tf.shape(decoder_seq))==3:
            decoder_seq=tf.squeeze(decoder_seq,axis=-1)
        encoder_padding_mask = self.create_padding_mask(encoder_seq)
        decoder_padding_mask = self.create_padding_mask(decoder_seq)
        return encoder_padding_mask[:, tf.newaxis] * decoder_padding_mask[:, :, tf.newaxis]


class BiasDeepNeuralNetworkLayer(tf.keras.layers.Layer):
    def __init__(self,position_bias_features={'position_feature':'position','page_feature':'page','neighbourhood_features':['near_expo_seq_cate2','near_expo_seq_cate3']},
                 feature_dims=160000, embedding_dims=16,max_position=400,max_page=100,neighbour_num=7):
        super(BiasDeepNeuralNetworkLayer, self).__init__()
        self.position_bias_features=position_bias_features
        self.neighbourhood_embedding = Embedding(input_dim=feature_dims, output_dim=embedding_dims)
        self.position_embedding=Embedding(input_dim=max_position, output_dim=embedding_dims)
        self.page_embedding = Embedding(input_dim=max_page, output_dim=embedding_dims)
        self.neighbour_num=neighbour_num
        self.embedding_dims=embedding_dims
        self.mlp=make_mlp_layer([8,1],'ReLU')

    def call(self,inputs):
        X_position = inputs[self.position_bias_features['position_feature']]
        if len(X_position.shape) == 2:
            X_position = tf.squeeze(X_position, axis=1)
        X_position = self.position_embedding(X_position)  # (batch_size,embedding_dim)

        X_page = inputs[self.position_bias_features['page_feature']]
        if len(X_page.shape) == 2:
            X_page = tf.squeeze(X_page, axis=1)
        X_page = self.page_embedding(X_page)              # (batch_size,embedding_dim)

        X_neighbour = tf.stack([inputs[feature] for feature in self.position_bias_features['neighbourhood_features']],
                            axis=1)  # (batch_size, features,neighbours)
        feature_num=len(self.position_bias_features['neighbourhood_features'])
        X_neighbour = tf.reshape(X_neighbour, [-1, feature_num * self.neighbour_num])  # (batch_size,feature_cols*seq_length)
        X_neighbour = self.neighbourhood_embedding(X_neighbour)  # (batch_size,feature_num*neighbour_num,emb_dim)
        X_neighbour = tf.reshape(X_neighbour, [-1, feature_num,self.neighbour_num,self.embedding_dims])

        query = Lambda(lambda x: x[:, :,3:4, :])(X_neighbour)
        attention_scores = tf.einsum('bhqe,bhke->bhqk',query,X_neighbour)
        attention_scores_scaled = Lambda(lambda x: x / (X_neighbour.shape[-1] ** 0.5))(attention_scores)
        attention_weights = Activation('softmax')(attention_scores_scaled)
        attention_output = tf.einsum('bhqk,bhke->bhqe',attention_weights,X_neighbour)
        attention_weights=tf.keras.layers.Flatten()(attention_weights)
        attention_output=tf.keras.layers.Flatten()(attention_output)
        result=tf.concat([attention_weights,attention_output,X_position,X_page],axis=1)
        result=self.mlp(result)

        return result


if __name__ == '__main__':
    inputs = {
        'uid': tf.constant([0, 1, 2]),
        'utag1': tf.constant([6, 7, 8]),
        'utag2': tf.constant([9, 10, 11]),
        'utag3': tf.constant([12, 13, 14]),
        'utag4': tf.constant([15, 16, 17]),
        'i_cate_id': tf.constant([11, 12, 13]),
        'i_goods_id': tf.constant([14, 15, 16]),
        'i_shop_id': tf.constant([17, 18, 19]),
        'continuous_feature1': tf.constant([0.2, 7.8, 4.9]),
        'continuous_feature2': tf.constant([5.3, 1.2, 8.0]),
        'continuous_feature3': tf.constant([-3.8, -19.6, 4.2]),
        'click_goods_ids': tf.constant([[30, 31, 32, 0, 0, 0], [33, 34, 35, 36, 0, 0], [37, 38, 39, 40, 41,0]]),
        'click_shop_ids': tf.constant([[42, 43, 44, 0, 0, 0], [45, 46, 47, 48, 0, 0], [49, 50, 51, 52, 53, 0]]),
        'click_cate_ids': tf.constant([[54, 55, 56, 0, 0, 0], [57, 58, 59, 60, 0, 0], [61, 62, 63, 64, 65, 0]]),
        'cart_goods_ids': tf.constant([[30, 31, 32, 0, 0, 0], [33, 34, 35, 36, 0, 0], [37, 38, 39, 40, 41,0]]),
        'cart_shop_ids': tf.constant([[42, 43, 44, 0, 0, 0], [45, 46, 47, 48, 0, 0], [49, 50, 51, 52, 53, 0]]),
        'cart_cate_ids': tf.constant([[54, 55, 56, 0, 0, 0], [57, 58, 59, 60, 0, 0], [61, 62, 63, 64, 65, 0]]),
        'order_goods_ids': tf.constant([[30, 31, 32, 0, 0, 0], [33, 34, 35, 36, 0, 0], [37, 38, 39, 40, 41, 0]]),
        'order_shop_ids': tf.constant([[42, 43, 44, 0, 0, 0], [45, 46, 47, 48, 0, 0], [49, 50, 51, 52, 53, 0]]),
        'order_cate_ids': tf.constant([[54, 55, 56, 0, 0, 0], [57, 58, 59, 60, 0, 0], [61, 62, 63, 64, 65, 0]]),
        'position': tf.constant([1, 8, 24]),
        'page': tf.constant([1, 2, 3]),
        'near_expo_seq_cate2':tf.constant([[0,0,0,100,101,102,103],[104,105,106,107,108,109,110],[111,112,113,114,115,116,117]]),
        'near_expo_seq_cate3': tf.constant(
            [[0, 0, 0, 200, 201, 202, 203], [204, 205, 206, 207, 208, 209, 210], [211, 212, 213, 214, 215, 216, 217]])
    }
    layer = DMTLayer()
    set_custom_initialization(layer)
    print(layer(inputs))
    print('****************')
    input_dict = {}
    max_length = 6
    neighbour_num=7
    for feature in layer.user_and_context_categorical_features + layer.item_categorical_features+[layer.position_bias_features['position_feature'],layer.position_bias_features['page_feature']]:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.continuous_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
    for feature in layer.click_series_features+layer.cart_series_features+layer.order_series_features:
        input_dict[feature] = tf.keras.Input(shape=(max_length,), name=feature, dtype=tf.int64)
    for feature in layer.position_bias_features['neighbourhood_features']:
        input_dict[feature] = tf.keras.Input(shape=(neighbour_num,), name=feature, dtype=tf.int64)
    output = layer(input_dict)
    model = tf.keras.Model(input_dict, output)
    model.summary()
    print(model(inputs))
