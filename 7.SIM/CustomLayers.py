import tensorflow as tf
from tensorflow.python.ops.math_ops import MatMul
from tensorflow.python.ops.nn_ops import BiasAdd
from tensorflow.python.keras import activations, initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization
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


class GSULayer(tf.keras.layers.Layer):
    def __init__(self,item_categorical_features=['i_goods_id', 'i_shop_id', 'i_cate_id'],
                 behavior_series_features=['visited_goods_ids', 'visited_shop_ids', 'visited_cate_ids'],
                 feature_dims=1000, embedding_dims=16,
                 activation='Dice', padding_index=0,embedding_layer=None,l2_reg=0.01):
        super(GSULayer, self).__init__()
        self.item_categorical_features = item_categorical_features
        assert len(item_categorical_features) == len(
            behavior_series_features), "Features to be interacted should match in item and behavior series"
        self.behavior_series_features = behavior_series_features
        self.feature_dims = feature_dims
        self.embedding_dims = embedding_dims

        if embedding_layer:
            self.embed=embedding_layer
        else:
            self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims,embeddings_regularizer=l2(l2_reg))

        self.mlp = make_mlp_layer([200, 80], activation=activation, softmax_units=2)

        self.padding_index = padding_index

    def build(self, input_shape):
        super().build(input_shape)

    def inner_product_attention(self,target_emb,series_emb,valid_mask):
        # target_emb (batch_size,embedding_dim)
        # series_emb (batch_size,sequence_length,embedding_dim)
        # valid_mask (batch_size,sequence_length)
        attentioned_scores=tf.einsum('be,ble->bl', target_emb, series_emb)
        masked_scores=attentioned_scores*tf.cast(valid_mask,tf.float32)
        pooled_embedding=tf.einsum('bl,ble->be', masked_scores, series_emb)
        return pooled_embedding

    def call(self, inputs):
        X_item = []
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
        X_series = tf.reshape(X_series, [-1, feature_cols * seq_length])  # (batch_size,feature_cols*seq_length)
        X_series = self.embed(X_series)  # (batch_size,feature_cols*seq_length,emb_dim)
        X_series = tf.reshape(X_series, [-1, seq_length,
                                         feature_cols * self.embedding_dims])  # (batch_size,seq_length,feature_cols*emb_dim)

        # (batch_size,seq_length)
        valid_mask = inputs[self.behavior_series_features[0]] != self.padding_index
        # (1)如果是训练历史全量序列数据，那么按照常规做法执行
        pooled_embedding=self.inner_product_attention(X_item,X_series,valid_mask)
        # (2)如果增量更新时，X_series应该是sequence_length为1的请求向量，代表新增的用户行为
        # pooled_embedding先从Redis或者其它方式，根据uid查得当前的累计pooled_embedding
        # pooled_embedding+=self.inner_product_attention(X_item,X_series,valid_mask)
        # 将新pooled_embedding存入到刚才读取的数据源中完成更新
        X_combined=tf.concat([X_item,pooled_embedding],axis=1)

        output = self.mlp(X_combined)
        result = {'output': output,'X_series':X_series,'valid_mask':valid_mask}
        return result


class ESULayer(tf.keras.layers.Layer):
    def __init__(self, user_and_context_categorical_features=['uid', 'utag1', 'utag2', 'utag3', 'utag4'],
                 item_categorical_features=['i_goods_id', 'i_shop_id', 'i_cate_id'],
                 behavior_series_features=['visited_goods_ids', 'visited_shop_ids', 'visited_cate_ids'],
                 feature_dims=1000, embedding_dims=16,
                 activation='Dice', padding_index=0,num_heads=8,embedding_layer=None,l2_reg=0.01):
        super(ESULayer, self).__init__()
        self.user_and_context_categorical_features = user_and_context_categorical_features
        self.item_categorical_features = item_categorical_features
        assert len(item_categorical_features) == len(
            behavior_series_features), "Features to be interacted should match in item and behavior series"
        self.behavior_series_features = behavior_series_features
        self.feature_dims = feature_dims
        self.embedding_dims = embedding_dims

        if embedding_layer:
            self.embed = embedding_layer
        else:
            self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                                   self.embedding_dims,embeddings_regularizer=l2(l2_reg))

        self.mlp = make_mlp_layer([200, 80], activation=activation, softmax_units=2)

        self.padding_index = padding_index

        self.dien_layer = DIENLayer(stage='inference')
        #self.dien_layer.load_weights('path_to_the_weights.h5')
        self.dien_layer.trainable = False

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=len(behavior_series_features)*embedding_dims)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        dien_output = self.dien_layer(inputs)['X_combined']

        X_item = []
        for feature in self.item_categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_item.append(feature_tensor)
        X_item = tf.concat(X_item, axis=1)
        X_item = tf.keras.layers.Flatten()(self.embed(X_item))
        query=tf.expand_dims(X_item,axis=1)

        """
        extracted_series = tf.stack([inputs[feature] for feature in self.behavior_series_features],
                            axis=2)  # (batch_size, seq_length, feature_cols)
        batch_size, seq_length, feature_cols = extracted_series.shape
        extracted_series = tf.reshape(extracted_series, [-1, feature_cols * seq_length])  # (batch_size,feature_cols*seq_length)
        extracted_series = self.embed(extracted_series)  # (batch_size,feature_cols*seq_length,emb_dim)
        extracted_series = tf.reshape(extracted_series, [-1, seq_length,
                                         feature_cols * self.embedding_dims])  # (batch_size,seq_length,feature_cols*emb_dim)

        series_mask = inputs[self.behavior_series_features[0]] == self.padding_index
        """
        extracted_series=inputs['extracted_series']
        series_mask=inputs['series_mask']

        value=key=extracted_series
        attention_mask=tf.expand_dims(series_mask,1)
        # (batch_size,1,embedding_dim*len(feature_cols)
        mha_interest = self.mha(query, value, key,
                                       attention_mask=attention_mask)
        mha_interest=tf.squeeze(mha_interest,axis=1)

        X_combined = tf.concat([dien_output,mha_interest], axis=1)
        output = self.mlp(X_combined)
        result = {'output': output}
        return result

class SIMLayer(tf.keras.layers.Layer):
    # 这里把内积算分数的逻辑重复了一遍，实际上可以从GSU层直接完成并传输结果
    # 或者加上其它综合的逻辑，包括hard search等，替换soft_index_search的逻辑把筛选的行为序列传到SIM中
    def __init__(self, user_and_context_categorical_features=['uid', 'utag1', 'utag2', 'utag3', 'utag4'],
                 item_categorical_features=['i_goods_id', 'i_shop_id', 'i_cate_id'],
                 behavior_series_features=['visited_goods_ids', 'visited_shop_ids', 'visited_cate_ids'],
                 feature_dims=1000, embedding_dims=16,
                 activation='Dice', padding_index=0,k=3,l2_reg=0.01):
        super(SIMLayer, self).__init__()
        assert len(item_categorical_features) == len(
            behavior_series_features), "Features to be interacted should match in item and behavior series"
        self.user_and_context_categorical_features=user_and_context_categorical_features
        self.item_categorical_features=item_categorical_features
        self.behavior_series_features=behavior_series_features

        self.embed = tf.keras.layers.Embedding(feature_dims,embedding_dims,embeddings_regularizer=l2(l2_reg))

        self.gsu_layer=GSULayer(item_categorical_features=item_categorical_features,
                 behavior_series_features=behavior_series_features,
                 feature_dims=feature_dims, embedding_dims=embedding_dims,
                 activation=activation, padding_index=padding_index,embedding_layer=self.embed)

        self.esu_layer = ESULayer(user_and_context_categorical_features=user_and_context_categorical_features,
                                  item_categorical_features=item_categorical_features,
                                  behavior_series_features=behavior_series_features,
                                  feature_dims=feature_dims, embedding_dims=embedding_dims,
                                  activation=activation, padding_index=padding_index, embedding_layer=self.embed)

        self.k=k

    def build(self, input_shape):
        super().build(input_shape)

    def soft_index_search(self, X_item, X_series, valid_mask):
        # X_item:     (batch_size, embedding_dim)
        # X_series:   (batch_size, sequence_length, embedding_dim)
        # valid_mask: (batch_size, sequence_length)

        # Compute inner products
        inner_products = tf.einsum('be,ble->bl', X_item, X_series)

        # Replace scores of invalid positions with -inf
        inner_products = tf.where(valid_mask, inner_products, -np.inf)

        # Find the indices of the top-k scores
        top_k_indices = tf.argsort(inner_products, axis=-1, direction='DESCENDING')[:, :self.k]

        # Gather the corresponding embeddings
        batch_size = tf.shape(X_series)[0]
        batch_indices = tf.reshape(tf.range(batch_size), (batch_size, 1))
        batch_indices = tf.tile(batch_indices, [1, self.k])
        indices = tf.stack([batch_indices, top_k_indices], axis=-1)
        top_k_embeddings = tf.gather_nd(X_series, indices)

        # Extract the valid mask for the selected embeddings
        extracted_valid_mask = tf.gather_nd(valid_mask, indices)

        return top_k_embeddings, extracted_valid_mask


    def call(self, inputs):
        gsu_output=self.gsu_layer(inputs)
        gsu_logits,X_series,valid_mask=gsu_output['output'],gsu_output['X_series'],gsu_output['valid_mask']

        X_item = []
        for feature in self.item_categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_item.append(feature_tensor)
        X_item = tf.concat(X_item, axis=1)
        X_item = tf.keras.layers.Flatten()(self.embed(X_item))

        top_k_embeddings, extracted_valid_mask=self.soft_index_search(X_item,X_series,valid_mask)
        inputs['extracted_series']=top_k_embeddings
        inputs['series_mask']=~extracted_valid_mask

        esu_logits = self.esu_layer(inputs)
        result={'gsu_logits':gsu_logits,'esu_logits':esu_logits}
        return result


class CoActionUnit(tf.keras.layers.Layer):
    """
    coaction_unit = CoActionUnit(induction_feature_num=20,
                                 feed_feature_num=20,
                                 feed_feature_dim=5,
                                 layer_unit_coef=[1, 1, 1],
                                 order=3,
                                 order_independent=True)
    induction_feature = tf.constant([1, 2, 3])
    feed_feature = tf.constant([4, 5, 6])
    output = coaction_unit([induction_feature, feed_feature])
    print(output)
    print('*******************')
    induction_feature = tf.constant([1, 2, 3])
    feed_feature = tf.constant([[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
    output = coaction_unit([induction_feature, feed_feature])
    print(output)
    """

    def __init__(self, induction_feature_num=1000, feed_feature_num=1000, feed_feature_dim=16, layer_unit_coef=None,
                 order=3,
                 order_independent=True, activation='tanh', padding_index=0):
        super(CoActionUnit, self).__init__()
        self.feed_embedding = tf.keras.layers.Embedding(feed_feature_num,
                                                        feed_feature_dim)
        if layer_unit_coef is None:
            layer_unit_coef = [1, 1, 1]
        num_layers = len(layer_unit_coef)
        self.W_indexes = []
        self.B_indexes = []
        self.W_shapes = []
        start = 0
        base = feed_feature_dim ** 2
        all_coefs = [1] + layer_unit_coef
        for i in range(num_layers):
            start_coef, end_coef = all_coefs[i], all_coefs[i + 1]
            self.W_shapes.append((-1, feed_feature_dim * start_coef, feed_feature_dim * end_coef))
            W_params = base * start_coef * end_coef
            B_params = feed_feature_dim * end_coef
            self.W_indexes.append([start, start + W_params])
            self.B_indexes.append([start + W_params, start + W_params + B_params])
            start = start + W_params + B_params
        induction_feature_dim = start
        self.induction_embedding = [tf.keras.layers.Embedding(induction_feature_num,
                                                              induction_feature_dim) for _ in
                                    range(order)] if order_independent else [
                                                                                tf.keras.layers.Embedding(
                                                                                    induction_feature_num,
                                                                                    induction_feature_dim)] * order  # 指向的是同一个层对象

        self.activation = [tf.keras.layers.Activation(activation) for _ in range(order)]

        self.padding_index = padding_index

    def call(self, inputs):
        induction_feature, feed_feature = inputs
        result = []  # 准备存放各阶feed的结果
        for i, embedding in enumerate(self.induction_embedding):
            order_result = self.basic_call((induction_feature, feed_feature, embedding, i + 1))
            result.append(order_result)
        return result

    def basic_call(self, inputs):
        # induction_feature :     (batch_size,)
        # feed_feature :         (batch_size,) 或者 (batch_size,feature_cnt)均可
        induction_feature, feed_feature, embedding, order = inputs
        if len(tf.shape(induction_feature))==2 and tf.shape(induction_feature)[1]==1:
            # Model通过input_dict读进来的特征是(batch_size,1)而不是(batch_size,)
            induction_feature=tf.squeeze(induction_feature,axis=1)
        feed_mask = (feed_feature == self.padding_index)
        feed_vector = self.feed_embedding(feed_feature) ** order
        # (batch_size,induction_feature_dim)
        induction_vector = embedding(induction_feature)
        W_matrixes = [induction_vector[:, start:end] for start, end in self.W_indexes]
        B_matrixes = [induction_vector[:, start:end] for start, end in self.B_indexes]
        feed_vector_shape = tf.shape(feed_vector)
        for W, B, W_shape, activation in zip(W_matrixes, B_matrixes, self.W_shapes, self.activation):
            W = tf.reshape(W, W_shape)
            # (batch_size,input_dim) ? (batch_size,input_dim,output_dim) + (batch_size,output_dim,)
            if len(feed_vector_shape) == 2:
                # (batch_size,1,input_dim)
                feed_vector = tf.expand_dims(feed_vector, 1)
                # (batch_size,output_dim)
                feed_vector = tf.squeeze(tf.matmul(feed_vector, W), 1) + B
            # (batch_size,feature_cnt,input_dim) @ (batch_size,input_dim,output_dim) + (batch_size,output_dim,)
            elif len(feed_vector_shape) == 3:
                B = tf.expand_dims(B, axis=1)
                feed_vector = tf.matmul(feed_vector, W) + B
            else:
                raise ValueError("feed_vector维度必须是2维或者3维")
            feed_vector = activation(feed_vector)
        feed_mask = tf.expand_dims(feed_mask, axis=-1)
        feed_vector = tf.where(feed_mask, 0., feed_vector)
        return feed_vector


class CANLayer(tf.keras.layers.Layer):
    def __init__(self, user_and_context_categorical_features=['uid', 'utag1', 'utag2', 'utag3', 'utag4'],
                 item_categorical_features=['i_goods_id', 'i_shop_id', 'i_cate_id'],
                 behavior_series_features=['visited_goods_ids', 'visited_shop_ids', 'visited_cate_ids'],
                 feature_dims=1000, embedding_dims=16,
                 activation='Dice', padding_index=0):
        super(CANLayer, self).__init__()
        self.user_and_context_categorical_features = user_and_context_categorical_features
        self.item_categorical_features = item_categorical_features
        assert len(item_categorical_features) == len(
            behavior_series_features), "Features to be interacted should match in item and behavior series"
        self.behavior_series_features = behavior_series_features
        self.feature_dims = feature_dims
        self.embedding_dims = embedding_dims

        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims)

        self.mlp = make_mlp_layer([200, 80], activation=activation, softmax_units=2)

        self.padding_index = padding_index

        self.dien_layer = DIENLayer(stage='inference')
        #self.dien_layer.load_weights('path_to_the_weights.h5')
        self.dien_layer.trainable = False

        self.CoActionUnit_dict = {}
        # 创建多少个CoActionUnit层，视乎想要如何独立地将特征交互。在一个独立的CoActionUnit层中，MLP embedding是独立的，feed embedding是独立的
        # 不独立的特征组合共享一个CoActionUnit层
        # 这里，i_goods_id,i_shop_id,i_cate_id分别使用不同的MLP embedding
        # 为了演示简单，i_goods_id只和visited_goods_ids与user_and_context_categorical_features交叉
        # (实际上可以和visited_shop_ids,visited_cate_ids交叉)，'i_shop_id', 'i_cate_id'同理
        for field_name in ['goods_id', 'shop_id', 'cate_id']:
            self.CoActionUnit_dict[field_name] = CoActionUnit(induction_feature_num=feature_dims,
                                                         feed_feature_num=feature_dims, feed_feature_dim=embedding_dims,
                                                         layer_unit_coef=None, order=3,
                                                         order_independent=True, activation='tanh', padding_index=0)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        dien_output = self.dien_layer(inputs)['X_combined']

        X_user = []
        for feature in self.user_and_context_categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_user.append(feature_tensor)
        X_user = tf.concat(X_user, axis=1)

        can_concated=[]
        for name in ['goods_id', 'shop_id', 'cate_id']:
            can_unit=self.CoActionUnit_dict[name]
            X_induction=inputs['i_'+name]
            X_series=inputs[f'visited_{name}s']
            user_output=can_unit([X_induction,X_user])
            user_output=tf.concat(user_output,axis=1)
            series_output=can_unit([X_induction,X_series])
            series_output=tf.concat(series_output,axis=1)
            series_output=tf.reduce_sum(series_output,axis=1,keepdims=True)
            can_concated.append(user_output)
            can_concated.append(series_output)

        X_flattened = tf.keras.layers.Flatten()(tf.concat(can_concated,axis=1))
        X_combined = tf.concat([dien_output,X_flattened], axis=1)
        output = self.mlp(X_combined)
        result = {'output': output}
        return result


class LSHAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_buckets=64, **kwargs):
        super(LSHAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.n_buckets = n_buckets
        self.projection_matrix = self.add_weight(
            name="projection_matrix",
            shape=(self.d_model, self.n_buckets // 2),
            initializer='random_normal',
            trainable=False
        )

    def call(self, inputs, mask=None):
        q, k, v = inputs
        batch_size, seq_len, _ = tf.shape(q)

        # Compute hash buckets
        buckets = self.hash_vectors(q)
        # Sort items by hash bucket
        idxs = tf.argsort(buckets, axis=-1)
        q = tf.gather(q, idxs, batch_dims=1)
        k = tf.gather(k, idxs, batch_dims=1)
        v = tf.gather(v, idxs, batch_dims=1)
        if mask is not None:
            mask = tf.gather(mask, idxs, batch_dims=1)
            mask = self.split_buckets(mask)
            print(mask.shape)

        # Split into buckets
        q, k, v = self.split_buckets(q), self.split_buckets(k), self.split_buckets(v)

        # Compute attention
        scores = tf.einsum('bhie,bhje->bhij', q, k) / np.sqrt(self.d_model)
        if mask is not None:
            scores = scores * mask - 1e9 * (1 - mask)
        attn = tf.nn.softmax(scores, axis=-1)
        output = tf.einsum('bhij,bhje->bhie', attn, v)

        # Concatenate all buckets
        output = tf.reshape(output, (batch_size, seq_len, -1))
        # Normalize the output
        output = output / tf.cast(self.n_buckets, dtype=tf.float32)

        return output

    def hash_vectors(self, x):
        # Project input vectors onto projection_matrix
        projected = tf.matmul(x, self.projection_matrix)
        # Use sign of projection to compute the hash
        hashes = tf.sign(projected)
        # Convert binary to integers for bucketing
        bucket_range = tf.constant(
            [2 ** i for i in range(self.n_buckets // 2)], dtype=tf.float32
        )
        bucket_ids = tf.reduce_sum(
            hashes * bucket_range, axis=-1
        )
        bucket_ids = tf.cast(bucket_ids, tf.int32) % self.n_buckets
        return bucket_ids

    def split_buckets(self, x):
        batch_size, seq_len = tf.shape(x)[:2]
        return tf.reshape(x, (batch_size, self.n_buckets, seq_len // self.n_buckets, -1))


class ETALayer(tf.keras.layers.Layer):
    def __init__(self, user_and_context_categorical_features=['uid', 'utag1', 'utag2', 'utag3', 'utag4'],
                 item_categorical_features=['label_goods_ids', 'label_shop_ids', 'label_cate_ids'],
                 longterm_series_features=['longterm_goods_ids', 'longterm_shop_ids', 'longterm_cate_ids'],
                 shortterm_series_features=['shortterm_goods_ids', 'shortterm_shop_ids', 'shortterm_cate_ids'],
                 feature_dims=160000, embedding_dims=16,
                 padding_index=0, num_heads=8,key_dim=4,
                 activation='ReLU',lsh_dim=16,lsh_topk=3):
        super(ETALayer, self).__init__()
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

        self.H = self.add_weight(
            name="projection_matrix",
            shape=(len(item_categorical_features)*embedding_dims, lsh_dim),
            initializer='random_normal',
            trainable=False
        )

        self.lsh_topk=lsh_topk

        self.shortterm_mha=tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.longterm_mha=tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

        self.final_mlp=make_mlp_layer([200, 80], activation=activation, sigmoid_units=True)

    def build(self, input_shape):
        super().build(input_shape)

    def generate_longterm_interest(self,item_embedding,longterm_series_embedding,longterm_mask):
        # 这里没有实现工程上把一个用户的所有候选物品放到一个batch，这时，longterm_series_embedding在batch内是相同的
        # series_codes=tf.sign(tf.einsum('le,em->lm',longterm_series_embedding,self.H))
        # item_codes=tf.sign(tf.einsum('be,em->bm',item_embedding,self.H))
        series_codes=tf.sign(tf.einsum('ble,em->blm',longterm_series_embedding,self.H))
        item_codes=tf.tile(tf.sign(tf.einsum('ble,em->blm',item_embedding,self.H)),[1,tf.shape(series_codes)[1],1])
        equalities = tf.cast(series_codes == item_codes, tf.float32)
        scores=tf.reduce_sum(equalities,axis=-1)      # batch_size,sequence_length
        scores=scores+tf.cast(longterm_mask,dtype=tf.float32)*(-np.inf)

        # 对scores进行排序并获得最大的k个索引
        sorted_indices = tf.argsort(scores, direction='DESCENDING')[:, :self.lsh_topk]

        # 使用gather根据索引收集longterm_series_embedding和longterm_mask
        top_longterm_series_embedding = tf.gather(longterm_series_embedding, sorted_indices, batch_dims=1)
        top_longterm_mask = tf.gather(longterm_mask, sorted_indices, batch_dims=1)

        sequence_lengths = tf.reduce_sum(tf.cast(~top_longterm_mask, tf.int32), axis=1)

        # 创建一个掩码矩阵，大小为batch_size*最大长度
        mask_matrix = tf.sequence_mask(sequence_lengths, top_longterm_mask.shape[1], dtype=tf.float32)
        attention_mask = tf.expand_dims(mask_matrix, 1)  # expand one dimension
        longterm_interest = self.longterm_mha(
            item_embedding, top_longterm_series_embedding, top_longterm_series_embedding,
            attention_mask=attention_mask)
        longterm_interest=tf.squeeze(longterm_interest,axis=1)
        return longterm_interest

    def generate_shorterm_interest(self,item_embedding,shortterm_series_embedding,shortterm_mask):
        sequence_lengths = tf.reduce_sum(tf.cast(~shortterm_mask, tf.int32), axis=1)

        # 创建一个掩码矩阵，大小为最大长度*最大长度
        mask_matrix = tf.sequence_mask(sequence_lengths, shortterm_mask.shape[1], dtype=tf.float32)
        attention_mask = tf.expand_dims(mask_matrix, 1)  # expand one dimension
        shortterm_interest = self.shortterm_mha(item_embedding, shortterm_series_embedding, shortterm_series_embedding,
                                   attention_mask=attention_mask)
        shortterm_interest = tf.squeeze(shortterm_interest, axis=1)
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
        shortterm_mask = inputs[self.shortterm_series_features[0]] == self.padding_index
        longterm_mask = inputs[self.longterm_series_features[0]] == self.padding_index

        X_item = self.stack_feature_embeddings(inputs,self.item_categorical_features)
        X_user = self.stack_feature_embeddings(inputs,self.user_and_context_categorical_features)
        X_short = self.stack_feature_embeddings(inputs,self.shortterm_series_features)
        X_long=self.stack_feature_embeddings(inputs,self.longterm_series_features)

        shortterm_insterest=self.generate_shorterm_interest(X_item,X_short,shortterm_mask)
        longterm_interest=self.generate_longterm_interest(X_item,X_long,longterm_mask)

        X_item=tf.squeeze(X_item,axis=1)
        X_user = tf.squeeze(X_user, axis=1)
        X_combined = tf.concat([X_item, X_user,shortterm_insterest,longterm_interest], axis=1)

        output = self.final_mlp(X_combined)

        result = {'output': output}
        return result



if __name__ == '__main__':
    """
    inputs = {
        'uid': tf.constant([0, 1, 2]),
        'utag1': tf.constant([6, 7, 8]),
        'utag2': tf.constant([9, 10, 11]),
        'utag3': tf.constant([12, 13, 14]),
        'utag4': tf.constant([60, 61, 62]),
        'i_cate_id': tf.constant([11, 12, 13]),
        'i_goods_id': tf.constant([14, 15, 16]),
        'i_shop_id': tf.constant([17, 18, 19]),
        'visited_goods_ids': tf.constant([[30, 31, 32, 0, 0, 0], [33, 34, 35, 36, 0, 0], [37, 38, 39, 40, 41, 0]]),
        'visited_shop_ids': tf.constant([[42, 43, 44, 0, 0, 0], [45, 46, 47, 48, 0, 0], [49, 50, 51, 52, 53, 0]]),
        'visited_cate_ids': tf.constant([[54, 55, 56, 0, 0, 0], [57, 58, 59, 60, 0, 0], [61, 62, 63, 64, 65, 0]]),
    }
    layer = GSULayer()
    print(layer(inputs))
    print('*******************')
    # 输入
    max_length = 6
    input_dic = {}
    for feature in layer.item_categorical_features:
        input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.behavior_series_features:  # DIN: for feature in layer.behavior_series_features:
        input_dic[feature] = tf.keras.Input(shape=(max_length,), name=feature, dtype=tf.int64)

    output = layer(input_dic)
    model = tf.keras.Model(input_dic, output)
    model.summary()
    print(model(inputs))
    """
    """
    inputs = {
        'uid': tf.constant([0, 1, 2]),
        'utag1': tf.constant([6, 7, 8]),
        'utag2': tf.constant([9, 10, 11]),
        'utag3': tf.constant([12, 13, 14]),
        'utag4': tf.constant([60, 61, 62]),
        'i_cate_id': tf.constant([11, 12, 13]),
        'i_goods_id': tf.constant([14, 15, 16]),
        'i_shop_id': tf.constant([17, 18, 19]),
        'visited_goods_ids': tf.constant([[30, 31, 32, 0, 0, 0], [33, 34, 35, 36, 0, 0], [37, 38, 39, 40, 41, 0]]),
        'visited_shop_ids': tf.constant([[42, 43, 44, 0, 0, 0], [45, 46, 47, 48, 0, 0], [49, 50, 51, 52, 53, 0]]),
        'visited_cate_ids': tf.constant([[54, 55, 56, 0, 0, 0], [57, 58, 59, 60, 0, 0], [61, 62, 63, 64, 65, 0]]),
    }
    layer = ESULayer()
    print(layer(inputs))
    print('*******************')
    # 输入
    max_length = 6
    input_dic = {}
    for feature in layer.item_categorical_features+layer.user_and_context_categorical_features:
        input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.behavior_series_features:  # DIN: for feature in layer.behavior_series_features:
        input_dic[feature] = tf.keras.Input(shape=(max_length,), name=feature, dtype=tf.int64)

    output = layer(input_dic)
    model = tf.keras.Model(input_dic, output)
    model.summary()
    print(model(inputs))
    """
    """
    def soft_index_search(X_item, X_series, valid_mask, k=2):
        # X_item:     (batch_size, embedding_dim)
        # X_series:   (batch_size, sequence_length, embedding_dim)
        # valid_mask: (batch_size, sequence_length)

        # Compute inner products
        inner_products = tf.einsum('be,ble->bl', X_item, X_series)

        # Replace scores of invalid positions with -inf
        inner_products = tf.where(valid_mask, inner_products, -np.inf)

        # Find the indices of the top-k scores
        top_k_indices = tf.argsort(inner_products, axis=-1, direction='DESCENDING')[:, :k]

        # Gather the corresponding embeddings
        batch_size = tf.shape(X_series)[0]
        batch_indices = tf.reshape(tf.range(batch_size), (batch_size, 1))
        batch_indices = tf.tile(batch_indices, [1, k])
        print(batch_indices)
        print(top_k_indices)
        indices = tf.stack([batch_indices, top_k_indices], axis=-1)
        top_k_embeddings = tf.gather_nd(X_series, indices)
        print(indices)
        print(X_series.shape)
        print(indices.shape)

        # Extract the valid mask for the selected embeddings
        extracted_valid_mask = tf.gather_nd(valid_mask, indices)

        return top_k_embeddings, extracted_valid_mask


    # Test the function
    batch_size = 3
    sequence_length = 6
    embedding_dim = 4
    k = 3

    X_item = tf.random.normal((batch_size, embedding_dim))
    X_series = tf.random.normal((batch_size, sequence_length, embedding_dim))
    valid_mask = tf.constant([[True, True, False, False, False, False],
                              [True, True, True, False, False, False],
                              [True, True, True, True, True, False]])

    top_k_embeddings, extracted_valid_mask = soft_index_search(X_item, X_series, valid_mask, k)
    print("Top-k embeddings:", top_k_embeddings)
    print("Extracted valid mask:", extracted_valid_mask)
    """
    """
    inputs = {
        'uid': tf.constant([0, 1, 2]),
        'utag1': tf.constant([6, 7, 8]),
        'utag2': tf.constant([9, 10, 11]),
        'utag3': tf.constant([12, 13, 14]),
        'utag4': tf.constant([60, 61, 62]),
        'i_cate_id': tf.constant([11, 12, 13]),
        'i_goods_id': tf.constant([14, 15, 16]),
        'i_shop_id': tf.constant([17, 18, 19]),
        'visited_goods_ids': tf.constant([[30, 31, 32, 0, 0, 0], [33, 34, 35, 36, 0, 0], [37, 38, 39, 40, 41, 0]]),
        'visited_shop_ids': tf.constant([[42, 43, 44, 0, 0, 0], [45, 46, 47, 48, 0, 0], [49, 50, 51, 52, 53, 0]]),
        'visited_cate_ids': tf.constant([[54, 55, 56, 0, 0, 0], [57, 58, 59, 60, 0, 0], [61, 62, 63, 64, 65, 0]]),
    }
    layer = SIMLayer()
    print(layer(inputs))
    print('*******************')
    # 输入
    max_length = 6
    input_dic = {}
    for feature in layer.item_categorical_features + layer.user_and_context_categorical_features:
        input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.behavior_series_features:  # DIN: for feature in layer.behavior_series_features:
        input_dic[feature] = tf.keras.Input(shape=(max_length,), name=feature, dtype=tf.int64)

    output = layer(input_dic)
    model = tf.keras.Model(input_dic, output)
    model.summary()
    print(model(inputs))
    """
    """
    inputs = {
        'uid': tf.constant([0, 1, 2]),
        'utag1': tf.constant([6, 7, 8]),
        'utag2': tf.constant([9, 10, 11]),
        'utag3': tf.constant([12, 13, 14]),
        'utag4': tf.constant([60, 61, 62]),
        'i_cate_id': tf.constant([11, 12, 13]),
        'i_goods_id': tf.constant([14, 15, 16]),
        'i_shop_id': tf.constant([17, 18, 19]),
        'visited_goods_ids': tf.constant([[30, 31, 32, 0, 0, 0], [33, 34, 35, 36, 0, 0], [37, 38, 39, 40, 41, 0]]),
        'visited_shop_ids': tf.constant([[42, 43, 44, 0, 0, 0], [45, 46, 47, 48, 0, 0], [49, 50, 51, 52, 53, 0]]),
        'visited_cate_ids': tf.constant([[54, 55, 56, 0, 0, 0], [57, 58, 59, 60, 0, 0], [61, 62, 63, 64, 65, 0]]),
    }
    layer = CANLayer()
    print(layer(inputs))
    print('*******************')
    # 输入
    max_length = 6
    input_dic = {}
    for feature in layer.user_and_context_categorical_features + layer.item_categorical_features:
        input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.behavior_series_features:  # DIN: for feature in layer.behavior_series_features:
        input_dic[feature] = tf.keras.Input(shape=(max_length,), name=feature, dtype=tf.int64)

    output = layer(input_dic)
    model = tf.keras.Model(input_dic, output)
    model.summary()
    print(model(inputs))
    """
    """
    # 定义一些模拟数据
    batch_size = 32
    seq_length = 128
    d_model = 64

    # 随机生成 Q, K, V
    q = tf.random.normal((batch_size, seq_length, d_model))
    k = tf.random.normal((batch_size, seq_length, d_model))
    v = tf.random.normal((batch_size, seq_length, d_model))

    # 生成模拟mask，其中，我们将部分序列的后半部分设为0，模拟padding部分
    mask = np.ones((batch_size, seq_length))
    for i in range(batch_size):
        pad_len = np.random.randint(0, seq_length // 2)  # 随机选择padding长度
        mask[i, -pad_len:] = 0
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)

    # 创建 LSH 注意力层并进行前向计算
    attention_layer = LSHAttention(d_model=d_model, n_buckets=64)
    output = attention_layer([q, k, v], mask=mask)

    print("Input Q shape:", q.shape)
    print("Output shape:", output.shape)

    # 验证输出形状与输入 Q 的形状相同
    assert q.shape == output.shape
    """
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
        'label_cate_ids': tf.constant([11, 12, 13]),
        'label_goods_ids': tf.constant([31, 32, 33]),
        'label_shop_ids': tf.constant([51, 52, 53])
    }
    user_and_context_categorical_features = ['uid', 'utag1', 'utag2', 'utag3', 'utag4'],
    item_categorical_features = ['label_goods_ids', 'label_shop_ids', 'label_cate_ids'],
    longterm_series_features = ['longterm_goods_ids', 'longterm_shop_ids', 'longterm_cate_ids'],
    shortterm_series_features = ['shortterm_goods_ids', 'shortterm_shop_ids', 'shortterm_cate_ids']

    layer = ETALayer()
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
