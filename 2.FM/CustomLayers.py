import tensorflow as tf
from tensorflow.python.ops.math_ops import MatMul
from tensorflow.python.ops.nn_ops import BiasAdd
from tensorflow.python.keras import activations, initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import Dropout

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
    def __init__(self,units,activation=None,use_bias=True,is_batch_norm=False,
                 is_dropput=0,kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',**kwargs):
        super(MLPLayer,self).__init__(**kwargs)

        self.units=[units] if not isinstance(units,list) else units
        if len(self.units) <= 0:
            raise ValueError(
                f'Received an invalid value for `units`, expected '
                f'a positive integer, got {units}.')
        self.use_bias=use_bias
        self.is_batch_norm=is_batch_norm
        self.is_dropout=is_dropput
        # 调用标准激活函数和初始化方式的常用api
        self.activation = activations.get(activation)
        self.kernel_initializer=initializers.get(kernel_initializer)
        self.bias_initializer=initializers.get(bias_initializer)

    def build(self,input_shape):
        input_shape=tensor_shape.TensorShape(input_shape)
        # 进入全连接层的输入维度
        last_dim=tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')

        dims=[last_dim]+self.units

        self.kernels=[]
        self.biases=[]
        self.bns=[]

        # 1层输入，len(dims)-1层全连接层及对应输出
        for i in range(len(dims)-1):
            # 第i层的全部参数
            self.kernels.append(
                self.add_weight(f'kernel_{i}',
                                shape=[dims[i],dims[i+1]],
                                initializer=self.kernel_initializer,
                                trainable=True))

            if self.use_bias:
                self.biases.append(
                    self.add_weight(f'bias_{i}',
                                    shape=[dims[i+1],],
                                    initializer=self.bias_initializer,
                                    trainable=True))

            self.bns.append(tf.keras.layers.BatchNormalization())
        self.built=True

    def call(self,inputs,is_train=False):
        _input=inputs
        for i in range(len(self.units)):
            _input=MatMul(a=_input,b=self.kernels[i])
            if self.use_bias:
                _input=BiasAdd(value=_input,bias=self.biases[i])
            if self.is_batch_norm:
                _input=self.bns[i](_input)
            if self.activation is not None:
                _input=self.activation(_input)
            if is_train and self.is_dropout>0:
                _input=Dropout(self.is_dropout)(_input)  # is_dropout是丢弃率，不是通过率
        return _input


class FMRankingLayer(tf.keras.layers.Layer):
    """mm=ModelManager(feature_names=['item_tag1', 'item_tag2', 'item_tag3'])
       input={'item_tag1':tf.constant([0,1,2]),'item_tag2':tf.constant([3,4,5]),'item_tag3':tf.constant([6,7,8])}
       print(mm.model(input))

      We recommend that descendants of `Layer` implement the following methods:

      * `__init__()`: Defines custom layer attributes, and creates layer state
        variables that do not depend on input shapes, using `add_weight()`.
      * `build(self, input_shape)`: This method can be used to create weights that
        depend on the shape(s) of the input(s), using `add_weight()`. `__call__()`
        will automatically build the layer (if it has not been built yet) by
        calling `build()`.
      * `call(self, inputs, *args, **kwargs)`: Called in `__call__` after making
        sure `build()` has been called. `call()` performs the logic of applying the
        layer to the input tensors (which should be passed in as argument).
        Two reserved keyword arguments you can optionally use in `call()` are:
          - `training` (boolean, whether the call is in inference mode or training
            mode). See more details in [the layer/model subclassing guide](
            https://www.tensorflow.org/guide/keras/custom_layers_and_models#privileged_training_argument_in_the_call_method)
          - `mask` (boolean tensor encoding masked timesteps in the input, used
            in RNN layers). See more details in [the layer/model subclassing guide](
            https://www.tensorflow.org/guide/keras/custom_layers_and_models#privileged_mask_argument_in_the_call_method)
        A typical signature for this method is `call(self, inputs)`, and user could
        optionally add `training` and `mask` if the layer need them. `*args` and
        `**kwargs` is only useful for future extension when more input parameters
        are planned to be added.
      * `get_config(self)`: Returns a dictionary containing the configuration used
        to initialize this layer. If the keys differ from the arguments
        in `__init__`, then override `from_config(self)` as well."""
    def __init__(self, feature_names=['item_tag1', 'item_tag2', 'item_tag3'],feature_dims=20, embedding_dims=16, **kwargs):
        super(FMRankingLayer, self).__init__(**kwargs)
        self.feature_names=feature_names
        self.feature_dims=feature_dims
        self.embedding_dims = embedding_dims

    def build(self, input_shape):
        # bias
        self.bias = self.add_weight(name='bias',
                                    shape=(1, ),
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
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X.append(feature_tensor)
        X = tf.concat(X, axis=1)

        w_output=self.w(X)
        emb_output=self.embed(X)

        first_order=tf.reduce_sum(w_output,axis=1)

        sum_of_square=tf.reduce_sum(tf.square(emb_output),axis=1)
        square_of_sum=tf.square(tf.reduce_sum(emb_output,axis=1))
        second_order=0.5*tf.reduce_sum(tf.subtract(square_of_sum,sum_of_square),axis=1,keepdims=True)

        output=tf.nn.sigmoid(self.bias+first_order+second_order)
        result={'output':output}
        return result

class DSSMSingleTowerLayer(tf.keras.layers.Layer):
    """
    全部特征embedding,展开,MLP成需要输出的向量维度
    mm = ModelManager(layer='dssm_single_tower', feature_names=['item_tag1', 'item_tag2', 'item_tag3'])
    input = {'item_tag1': tf.constant([0, 1, 2,3]), 'item_tag2': tf.constant([ 4, 5,6,7]),
             'item_tag3': tf.constant([ 8,9,10,11])}
    print(mm.model(input))
    """
    def __init__(self, feature_names=['item_tag1','item_tag2','item_tag3'],feature_dims=20, embedding_dims=8,mlp_dims=[64,32],final_dim=8, **kwargs):
        super(DSSMSingleTowerLayer, self).__init__(**kwargs)
        self.feature_names=feature_names
        self.feature_dims=feature_dims
        self.embedding_dims = embedding_dims
        self.mlp_dims=mlp_dims
        self.final_dim=final_dim

    def build(self, input_shape):
        self.embed=tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims,
                                               embeddings_regularizer="l2")
        self.mlp=MLPLayer(units=self.mlp_dims, activation='relu')
        self.final=MLPLayer(units=[self.final_dim], activation=None)
        self.built = True

    def call(self, inputs):
        X=[]
        for feature in self.feature_names:
            X.append(inputs[feature])

        try:
            X = tf.concat(X, axis=1)
        except wrapError:
            # 调用layer和调用model时输入包装的方式很不一样，以下方法是通过实验纠正得到正确的shape
            X = [X]
            X = tf.concat(X, axis=1)
            X = tf.transpose(X,[1,0])

        X=self.embed(X)
        # print(X==tf.concat(X, axis=1)) 全部为True
        X=tf.keras.layers.Flatten()(tf.concat(X, axis=1))

        X=self.mlp(X)
        X=self.final(X)

        result={'user_id':inputs.get('user_id',None),
                'item_id':inputs.get('item_id',None),
                'output':X}
        return result

class DSSMTwoTowerRetrievalLayer(tf.keras.layers.Layer):
    """
    考虑到单塔排序时的数据处理需求，数据做labelencode的时候用户特征和商品特征共用了一套index共feature_dims个，
    为了不重新做一遍labelencode，双塔召回时两个塔的embeddding_dim还是预留feature_dims个，实际上恰好有一半
    初始化后不会被更新。实际上如果单独开发双塔召回的话，也可以把labelencode这个过程对用户特征和商品特征各做一遍
    mm = ModelManager(layer='dssm_double_tower')
    input = {'user_tag1':tf.constant([12,13,14,15]),'user_tag2':tf.constant([16,17,18,19]),
             'item_tag1': tf.constant([0, 1, 2,3]), 'item_tag2': tf.constant([4, 5, 6,7]),
             'item_tag3': tf.constant([8,9,10,11])}
    print(mm.model(input))
    """

    def __init__(self,u_feature_names=['user_tag1','user_tag2'],i_feature_names=['item_tag1','item_tag2','item_tag3'],
                 u_feature_dims=20,i_feature_dims=20,u_embedding_dims=8,i_embedding_dims=8,u_mlp_dims=[64,32],i_mlp_dims=[64,32],
                 final_dim=8,**kwargs):
        super(DSSMTwoTowerRetrievalLayer, self).__init__(**kwargs)
        self.u_tower=DSSMSingleTowerLayer(feature_names=u_feature_names,feature_dims=u_feature_dims,embedding_dims=u_embedding_dims,mlp_dims=u_mlp_dims,final_dim=final_dim)
        self.i_tower=DSSMSingleTowerLayer(feature_names=i_feature_names,feature_dims=i_feature_dims,embedding_dims=i_embedding_dims,mlp_dims=i_mlp_dims,final_dim=final_dim)

    def build(self, input_shape):
        self.built=True

    def call(self,inputs):
        u_embedding=self.u_tower(inputs)['output']
        i_embedding=self.i_tower(inputs)['output']
        similarity = tf.keras.losses.cosine_similarity(u_embedding, i_embedding,axis=1)
        similarity = (1 + similarity) / 2

        result = {'user_embedding':u_embedding,
                  'item_embedding':i_embedding,
                  'output': similarity}
        return result

class DeepFMRankingLayer(tf.keras.layers.Layer):
    """
    input = {'user_tag0': tf.constant([12, 13, 14, 15]), 'user_tag1': tf.constant([16, 17, 18, 19]),
             'item_tag1': tf.constant([0, 1, 2, 3]), 'item_tag2': tf.constant([4, 5, 6, 7]),
             'item_tag3': tf.constant([8, 9, 10, 11])}
    input_dic = {}
    layer = DeepFMRankingLayer(embedding_dims=8)
    for feature in layer.feature_names:
        input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    output = layer(input_dic)
    model = tf.keras.Model(input_dic, output)
    model.summary()
    print(model(input))
    """
    def __init__(self, feature_names=['user_tag0','user_tag1','item_tag1', 'item_tag2', 'item_tag3'],feature_dims=20, embedding_dims=16,
                 mlp_dims=[32,8], **kwargs):
        super(DeepFMRankingLayer, self).__init__(**kwargs)
        self.feature_names=feature_names
        self.feature_dims=feature_dims
        self.embedding_dims = embedding_dims
        self.mlp_dims=mlp_dims

    def build(self, input_shape):
        # bias
        self.bias = self.add_weight(name='bias',
                                    shape=(1, ),
                                    trainable=True)
        # embedding and w
        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims,
                                               embeddings_regularizer="l2")
        self.w = tf.keras.layers.Embedding(self.feature_dims,
                                           1,
                                           embeddings_regularizer="l2")
        self.MLP_layer1=MLPLayer(units=self.mlp_dims, activation='relu')
        self.MLP_layer2=MLPLayer(units=[1])
        self.built = True

    def call(self, inputs):
        X = []
        for feature in self.feature_names:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X.append(feature_tensor)
        X = tf.concat(X, axis=1)

        # common input vectors
        w_output=self.w(X)
        emb_output=self.embed(X)

        # FM part
        first_order=tf.add(tf.reduce_sum(w_output,axis=1),self.bias)
        sum_of_square=tf.reduce_sum(tf.square(emb_output),axis=1)
        square_of_sum=tf.square(tf.reduce_sum(emb_output,axis=1))
        second_order=0.5*tf.reduce_sum(tf.subtract(square_of_sum,sum_of_square),axis=1,keepdims=True)
        fm_part =first_order+second_order

        # DNN Part
        dense_embedding=tf.keras.layers.Flatten()(emb_output)
        dnn_part=self.MLP_layer2(self.MLP_layer1(dense_embedding))

        # output
        # 原论文的FM part没有接入全连接层，而是直接以系数1加入结果，这里也通过全连接层学习系数k
        output=tf.nn.sigmoid(fm_part+dnn_part)

        result={'output':output}
        return result

class WideAndDeepRankingLayer(tf.keras.layers.Layer):
    """
    mm = ModelManager(layer='wide_and_deep')
    input = {'user_tag1':tf.constant([12,13,14,15]),'user_tag2':tf.constant([16,17,18,19]),
             'item_tag1': tf.constant([0, 1, 2,3]), 'item_tag2': tf.constant([4, 5, 6,7]),
             'item_tag3': tf.constant([8,9,10,11])}
    print(mm.model(input))
    """
    def __init__(self, feature_names=['user_tag0','user_tag1','item_tag1', 'item_tag2', 'item_tag3'],feature_dims=20, embedding_dims=16,
                 mlp_dims=[32,8], **kwargs):
        super(WideAndDeepRankingLayer, self).__init__(**kwargs)
        self.feature_names=feature_names
        self.feature_dims=feature_dims
        self.embedding_dims = embedding_dims
        self.mlp_dims=mlp_dims
        raise ValueError("需要准备含labelencode以前原始特征值的数据集，本次工程不适用")

    def build(self, input_shape):
        # bias
        self.bias = self.add_weight(name='bias',
                                    shape=(1, ),
                                    trainable=True)
        # embedding and w
        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims,
                                               embeddings_regularizer="l2")
        self.w = tf.keras.layers.Embedding(self.feature_dims,
                                           1,
                                           embeddings_regularizer="l2")
        self.MLP_layer1=MLPLayer(units=self.mlp_dims, activation='relu')
        self.MLP_layer2=MLPLayer(units=[1], activation='sigmoid')
        self.built = True

    def call(self, inputs):
        # 需要有labelencode以前的原始特征值才行，dataset需要另外处理
        X = []
        for feature in self.feature_names:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X.append(feature_tensor)
        X = tf.concat(X, axis=1)
        raw=inputs['raw']

        # common input vectors
        w_output=self.w(X)
        emb_output=self.embed(X)

        # DNN Part
        dense_embedding=tf.keras.layers.Flatten()(emb_output)
        dnn_vector=self.MLP_layer1(dense_embedding)

        # output
        combined_vector=tf.concat([raw,dnn_vector],axis=1)
        output=self.MLP_layer2(combined_vector)

        result={'output':output}
        return result


class FFMRankingLayer(tf.keras.layers.Layer):
    """
    mm=ModelManager(feature_names=['item_tag1', 'item_tag2', 'item_tag3'],layer='ffm_ranking)
    input={'item_tag1':tf.constant([0,1,2,3]),'item_tag2':tf.constant([4,5,6,7]),'item_tag3':tf.constant([8,9,10,11])}
    print(mm.model(input))
    """
    def __init__(self, feature_names=['item_tag1', 'item_tag2', 'item_tag3'],feature_dims=20, embedding_dims=16, **kwargs):
        super(FFMRankingLayer, self).__init__(**kwargs)
        self.feature_names=feature_names
        self.feature_dims=feature_dims
        self.fields_cnt=len(self.feature_names)
        self.embedding_dims = embedding_dims

    def build(self, input_shape):
        # bias
        self.bias = self.add_weight(name='bias',
                                    shape=(1, ),
                                    trainable=True)
        # w
        self.w = tf.keras.layers.Embedding(self.feature_dims,
                                           1,
                                           embeddings_regularizer="l2")
        # embedding
        self.embedding_list = [tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims,
                                               embeddings_regularizer="l2") for _ in range(self.fields_cnt)]

        self.built = True

    def call(self, inputs):
        X = []
        for feature in self.feature_names:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X.append(feature_tensor)
        X = tf.concat(X, axis=1)

        w_output=self.w(X)

        first_order=tf.reduce_sum(w_output,axis=1)

        ebd_out=[self.embedding_list[i](X) for i in range(self.fields_cnt)]    # field_cnt*(batch_size,fields_cnt,embedding_dim)
        interactions=[]
        for i in range(self.fields_cnt):
            for j in range(i+1,self.fields_cnt):
                interactions.append(tf.multiply(ebd_out[i][:,j],ebd_out[j][:,i]))
                # ebd_out[i][:,j]:第i个lookup_table的X全embedding中，选出第j个field的embedding (i域对j域专用)
                # 第i个lookup_table:专门用来和域i交互的全域embedding信息
                # FFM:给每个域单独配置一个全域embedding，专门用于和全域(其它域)交互时提取使用
        interactions=tf.stack(interactions,axis=1)
        second_order=tf.reduce_sum(tf.reduce_sum(interactions,axis=1),axis=1,keepdims=True)

        output=tf.nn.sigmoid(self.bias+first_order+second_order)
        result={'output':output}
        return result


class FieldAwareInteractionLayer(tf.keras.layers.Layer):
    # 不使用for循环
    def __init__(self, fields_cnt, feature_dims=20, embedding_dims=16, **kwargs):
        super(FieldAwareInteractionLayer,self).__init__(**kwargs)
        self.embedding_lookup_table = self.add_weight(name='v', shape=(feature_dims, fields_cnt, embedding_dims),
                                 trainable=True)

    def call(self, X):
        # X:(batch_size,fields_num)
        # Field-aware Interaction term
        embeddings = tf.nn.embedding_lookup(self.embedding_lookup_table, X)  # (batch_size,fields_cnt,fields_cnt,embedding_dim)
        embeddings_T = tf.transpose(embeddings, [0, 2, 1, 3])
        interactions = embeddings * embeddings_T  # a域对b域交互时的embedding * b域对a域交互时的embedding

        # 获取interactions的形状
        shape = tf.shape(interactions)
        batch_size, num_fields, embedding_dim = shape[0], shape[1], shape[3]

        # 创建一个上三角掩码
        mask = tf.linalg.band_part(tf.ones((num_fields, num_fields)), 0, -1)
        mask = mask - tf.linalg.band_part(tf.ones((num_fields, num_fields)), 0, 0)
        mask = tf.expand_dims(mask, 0)
        mask = tf.expand_dims(mask, -1)
        mask = tf.tile(mask, [batch_size, 1, 1, embedding_dim])  # 广播到与interactions相同的形状

        # 将掩码应用于interactions，以保留上三角部分
        interactions_triangle = tf.multiply(interactions, mask)

        # 使用tf.boolean_mask来提取上三角部分
        interactions_triangle_flat = tf.boolean_mask(interactions_triangle, mask > 0)

        # 重新整形为所需形状
        final_shape = (batch_size, num_fields * (num_fields - 1) // 2, embedding_dim)
        interactions_final = tf.reshape(interactions_triangle_flat, final_shape)
        return interactions_final


class FFMLayer(tf.keras.layers.Layer):
    # 不使用python for循环逻辑，直接在tf2向量中完成交叉
    def __init__(self, feature_names=['item_tag1', 'item_tag2', 'item_tag3','user_tag0','user_tag1'], feature_dims=20, embedding_dims=16, **kwargs):
        super(FFMLayer, self).__init__(**kwargs)
        self.feature_names = feature_names
        self.feature_dims = feature_dims
        self.fields_cnt = len(self.feature_names)
        self.embedding_dims = embedding_dims

    def build(self, input_shape):
        self.bias = self.add_weight(name='bias', shape=(1,), trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_dims, 1), trainable=True)
        self.fa_interaction_layer=FieldAwareInteractionLayer(self.fields_cnt)
        self.built = True

    def call(self, inputs):
        X = []
        for feature in self.feature_names:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X.append(feature_tensor)
        X = tf.concat(X, axis=1)

        # Linear term
        linear_term = tf.reduce_sum(tf.nn.embedding_lookup(self.w, X), axis=1)  # (batch_size,fields_cnt,1)  -> (batch_size,1)

        interaction_vectors=self.fa_interaction_layer(X)
        interaction_term=tf.reduce_sum(tf.reduce_sum(interaction_vectors,axis=1),axis=1,keepdims=True)

        output = tf.nn.sigmoid(self.bias+linear_term+interaction_term)
        result = {'output': output}
        return result


class FwFMLayer(tf.keras.layers.Layer):
    # 不使用python for循环逻辑，直接在tf2向量中完成交叉
    def __init__(self, feature_names=['item_tag1', 'item_tag2', 'item_tag3','user_tag0','user_tag1'], feature_dims=20, embedding_dims=16, **kwargs):
        super(FwFMLayer, self).__init__(**kwargs)
        self.feature_names = feature_names
        self.feature_dims = feature_dims
        self.fields_cnt = len(self.feature_names)
        self.embedding_dims = embedding_dims
        self.interaction_weights=tf.keras.layers.Dense(1)

    def build(self, input_shape):
        self.bias = self.add_weight(name='bias', shape=(1,), trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_dims, 1), trainable=True)
        self.fa_interaction_layer=FieldAwareInteractionLayer(self.fields_cnt)
        self.built = True

    def call(self, inputs):
        X = []
        for feature in self.feature_names:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X.append(feature_tensor)
        X = tf.concat(X, axis=1)

        # Linear term
        linear_term = tf.reduce_sum(tf.nn.embedding_lookup(self.w, X), axis=1)  # (batch_size,fields_cnt,1)  -> (batch_size,1)

        interaction_vectors=self.fa_interaction_layer(X)
        interaction_term=self.interaction_weights(tf.reduce_sum(interaction_vectors,axis=-1))

        output = tf.nn.sigmoid(self.bias+linear_term+interaction_term)
        result = {'output': output}
        return result


class PNNRankingLayer(tf.keras.layers.Layer):
    """
     mm = ModelManager(layer='pnn_ranking',model_params={'method':'outer','kernel':'vec'})
    input = {'user_tag1':tf.constant([12,13,14,15]),'user_tag2':tf.constant([16,17,18,19]),
             'item_tag1': tf.constant([0, 1, 2,3]), 'item_tag2': tf.constant([4, 5, 6,7]),
             'item_tag3': tf.constant([8,9,10,11])}
    print(mm.model(input))
    """
    def __init__(self, feature_names=['user_tag0','user_tag1','item_tag1', 'item_tag2', 'item_tag3'],feature_dims=20, embedding_dims=16,
                 mlp_dims=[32,8],dropout=0, method='inner',kernel_type=None,**kwargs):
        super(PNNRankingLayer, self).__init__(**kwargs)
        assert method in ('inner','outer')
        self.feature_names=feature_names
        self.feature_dims=feature_dims
        self.fields_cnt=len(self.feature_names)
        self.embedding_dims = embedding_dims
        self.mlp_dims=mlp_dims
        self.method=method
        self.dropout=dropout
        self.kernel_type=kernel_type

        # self.embedding_flatten_dims=self.fields_cnt*self.embedding_dims
        # self.nlp_input_dims=self.embedding_flatten_dims + (self.fields_cnt * (self.fields_cnt - 1)) // 2

    def build(self, input_shape):
        # bias
        if self.method=='inner':
            self.pn=InnerProductNetwork()
        elif self.method=='outer':
            self.pn=OuterProductNetwork(self.fields_cnt,self.embedding_dims,self.kernel_type)

        # embedding and w
        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims,
                                               embeddings_regularizer="l2")

        self.MLP_layer1=MLPLayer(units=self.mlp_dims, activation='relu',is_dropput=self.dropout)
        self.MLP_layer2=MLPLayer(units=[1], activation='sigmoid')
        self.built = True

    def call(self, inputs):
        X = []
        for feature in self.feature_names:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X.append(feature_tensor)
        X = tf.concat(X, axis=1)

        # common input vectors
        emb_output=self.embed(X)            # (batch_size,fields_cnt,embedding_size)

        product_output=self.pn(emb_output)  # (batch_size,fields_cnt*(fields_cnt- 1) // 2)
        dense_embedding=tf.keras.layers.Flatten()(emb_output) # (batch_size,fields_cnt*embedding_dims)

        combined_vector=tf.concat([dense_embedding,product_output],axis=1) # (batch_size,fields_cnt*embedding_dims+fields_cnt*(fields_cnt- 1) // 2)

        # output
        # 原论文的FM part没有接入全连接层，而是直接以系数1加入结果，这里也通过全连接层学习系数k
        output = self.MLP_layer1(combined_vector)
        output = self.MLP_layer2(output)

        result={'output':output}
        return result

class InnerProductNetwork(tf.keras.layers.Layer):
    """
    ipn=InnerProductNetwork()
    input=tf.constant(np.arange(24).reshape(2,3,4),dtype=float)
    print(input)
    print(ipn(input))
    """
    def __init__(self,**kwargs):
        super(InnerProductNetwork, self).__init__(**kwargs)

    def build(self,input_shape):
        self.built=True

    def call(self, x):
        num_fields=x.shape[1]
        # row, col = [], []
        products=[]
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                # row.append(i), col.append(j)
                products.append(tf.multiply(x[:,i,:],x[:,j,:]))
        stacked_product=tf.stack(products,axis=1)
        result = tf.reduce_sum(stacked_product, axis=2)
        return result


class OuterProductNetwork(tf.keras.layers.Layer):
    """
    opn=OuterProductNetwork(3,4,'mat')
    input=tf.constant(np.arange(24).reshape(2,3,4),dtype=float)
    print(input)
    print(opn(input))
    """
    def __init__(self, fields_cnt, embedding_dims, kernel_type=None,**kwargs):
        super(OuterProductNetwork,self).__init__(**kwargs)
        if not kernel_type:
            kernel_type='mat'
        assert kernel_type in ('mat','vec','num')
        product_rows_num = fields_cnt * (fields_cnt - 1) // 2
        if kernel_type == 'mat':
            self.kernel_shape = embedding_dims, product_rows_num, embedding_dims
        elif kernel_type == 'vec':
            self.kernel_shape = product_rows_num, embedding_dims
        elif kernel_type == 'num':
            self.kernel_shape = product_rows_num, 1

        self.kernel_type = kernel_type


    def build(self, input_shape):
        super(OuterProductNetwork,self).build(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.kernel_shape),
                                      initializer=initializers.get('random_normal'),
                                      trainable=True)
        self.built = True

    def call(self, x):
        num_fields=x.shape[1]
        # row, col = [], []
        products=[]
        if self.kernel_type!='mat':
            for i in range(num_fields - 1):
                for j in range(i + 1, num_fields):
                    # row.append(i), col.append(j)
                    products.append(tf.multiply(x[:,i,:],x[:,j,:]))
            stacked_product=tf.stack(products,axis=1)
            kerneled_product=tf.multiply(stacked_product,tf.expand_dims(self.kernel,0))
            result = tf.reduce_sum(kerneled_product, axis=2)
        else:
            p=[]
            q=[]
            for i in range(num_fields - 1):
                for j in range(i + 1, num_fields):
                    p.append(x[:,i,:])
                    q.append(x[:,j,:])
            p=tf.stack(p,axis=1)
            q=tf.stack(q,axis=1)
            kp=tf.multiply(tf.expand_dims(p,1),self.kernel)
            kp = tf.transpose(tf.reduce_sum(kp, axis=-1),perm=[0,2,1])
            result=tf.reduce_sum(tf.multiply(kp,q,),axis=-1)
        return result


class PNNLayer(tf.keras.layers.Layer):
    """
    新版实现，改成不使用for循环
    input = {'item_tag1': tf.constant([0, 1, 2, 3]), 'item_tag2': tf.constant([4, 5, 6, 7]),
             'item_tag3': tf.constant([8, 9, 10, 11]),'user_tag0': tf.constant([12, 13, 14, 15]),
             'user_tag1': tf.constant([16, 17, 18, 19])}
    input_dic = {}
    layer = PNNLayer(embedding_dims=8,method='outer',kernel_type='num')
    print(layer(input))
    for feature in layer.feature_names:
        input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    output = layer(input_dic)
    model = tf.keras.Model(input_dic, output)
    model.summary()
    print(model(input))
    """
    def __init__(self, feature_names=['user_tag0','user_tag1','item_tag1', 'item_tag2', 'item_tag3'],feature_dims=20, embedding_dims=16,
                 mlp_dims=[32,8],dropout=0, method='inner',kernel_type=None,**kwargs):
        super(PNNLayer, self).__init__(**kwargs)
        assert method in ('inner','outer')
        self.feature_names=feature_names
        self.feature_dims=feature_dims
        self.fields_cnt=len(self.feature_names)
        self.embedding_dims = embedding_dims
        self.mlp_dims=mlp_dims
        self.method=method
        self.dropout=dropout
        self.kernel_type=kernel_type

    def build(self, input_shape):
        # bias
        if self.method=='inner':
            self.pn=IpnLayer()
        elif self.method=='outer':
            self.pn=OpnLayer(self.fields_cnt,self.embedding_dims,self.kernel_type)

        # embedding and w
        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims,
                                               embeddings_regularizer="l2")

        self.MLP_layer1=MLPLayer(units=self.mlp_dims, activation='relu',is_dropput=self.dropout)
        self.MLP_layer2=MLPLayer(units=[1], activation='sigmoid')
        self.built = True

    def call(self, inputs):
        X = []
        for feature in self.feature_names:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X.append(feature_tensor)
        X = tf.concat(X, axis=1)

        # common input vectors
        emb_output=self.embed(X)            # (batch_size,fields_cnt,embedding_size)

        product_output=self.pn(emb_output)  # (batch_size,fields_cnt*(fields_cnt- 1) // 2)
        dense_embedding=tf.keras.layers.Flatten()(emb_output) # (batch_size,fields_cnt*embedding_dims)

        combined_vector=tf.concat([dense_embedding,product_output],axis=1) # (batch_size,fields_cnt*embedding_dims+fields_cnt*(fields_cnt- 1) // 2)

        # output
        # 原论文的FM part没有接入全连接层，而是直接以系数1加入结果，这里也通过全连接层学习系数k
        output = self.MLP_layer1(combined_vector)
        output = self.MLP_layer2(output)

        result={'output':output}
        return result

def SharedFieldsInteraction(x):
    shape=tf.shape(x)
    batch_size = shape[0]
    num_fields = shape[1]
    embedding_dim = shape[2]
    x1 = tf.expand_dims(x, axis=1)
    x2 = tf.expand_dims(x, axis=2)
    interactions = tf.multiply(x1, x2)  # (batch_size,num_fields,num_fields,embedding_dim)
    ##### 对角处理逻辑开始,适合于域交叉剔重和排除自己交叉   input: num_fields   interactions (batch_size,num_fields,num_fields,embedding_dim)
    mask = tf.linalg.band_part(tf.ones((num_fields, num_fields)), 0, -1) - tf.linalg.band_part(
        tf.ones((num_fields, num_fields)), 0, 0)
    mask = tf.expand_dims(tf.expand_dims(mask, 0), -1)
    mask = tf.tile(mask, [batch_size, 1, 1, embedding_dim])  # 广播到与interactions相同的形状
    interactions_triangle = tf.multiply(interactions, mask)
    interactions_triangle_flat = tf.boolean_mask(interactions_triangle, mask > 0)
    final_shape = (batch_size, num_fields * (num_fields - 1) // 2, embedding_dim)
    interaction_final = tf.reshape(interactions_triangle_flat, final_shape)
    return interaction_final


class IpnLayer(tf.keras.layers.Layer):
    """
    不使用for循环
    input=tf.random.normal(3,6,8)
    layer=IpnLayer()
    print(layer(input))
    """
    def __init__(self,**kwargs):
        super(IpnLayer, self).__init__(**kwargs)

    def build(self,input_shape):
        self.built=True

    def call(self, x):
        # 交叉完成后，不合并域，而是将embedding维累加
        interaction_final=SharedFieldsInteraction(x)
        result = tf.reduce_sum(interaction_final, axis=2)
        return result


class OpnLayer(tf.keras.layers.Layer):
    """
    opn=OuterProductNetwork(6,8,'mat')
    input=tf.constant(np.arange(24).reshape(2,3,4),dtype=float)
    print(input)
    print(opn(input))
    """
    def __init__(self, fields_cnt, embedding_dims, kernel_type=None,**kwargs):
        super(OpnLayer,self).__init__(**kwargs)
        if not kernel_type:
            kernel_type='mat'
        assert kernel_type in ('mat','vec','num')
        product_rows_num = fields_cnt * (fields_cnt - 1) // 2
        if kernel_type == 'mat':
            self.kernel_shape = embedding_dims, product_rows_num, embedding_dims
        elif kernel_type == 'vec':
            self.kernel_shape = product_rows_num, embedding_dims
        elif kernel_type == 'num':
            self.kernel_shape = product_rows_num, 1

        self.kernel_type = kernel_type


    def build(self, input_shape):
        super(OpnLayer,self).build(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.kernel_shape),
                                      initializer=initializers.get('random_normal'),
                                      trainable=True)
        self.built = True

    def call(self, x):
        num_fields=x.shape[1]
        if self.kernel_type!='mat':
            stacked_product=SharedFieldsInteraction(x)
            kerneled_product=tf.multiply(stacked_product,tf.expand_dims(self.kernel,0))
            result = tf.reduce_sum(kerneled_product, axis=2)
        else:
            i, j = tf.meshgrid(tf.range(num_fields), tf.range(num_fields), indexing='ij')
            mask_upper_triangle = tf.cast(i < j, tf.int32)
            p = tf.boolean_mask(i, mask_upper_triangle)
            q = tf.boolean_mask(j, mask_upper_triangle)

            batch_size = tf.shape(x)[0]
            batch_indices = tf.reshape(tf.range(batch_size), (batch_size, 1))
            batch_indices = tf.tile(batch_indices, [1, self.kernel_shape[1]])  # product_rows_num
            indices_p = tf.stack([batch_indices, tf.broadcast_to(p, tf.shape(batch_indices))], axis=-1)
            indices_q = tf.stack([batch_indices, tf.broadcast_to(q, tf.shape(batch_indices))], axis=-1)
            X_p = tf.gather_nd(x, indices_p)
            X_q = tf.gather_nd(x, indices_q)
            # kernel :(embedding_dims, product_rows_num, embedding_dims)
            # (batch_size,1,product_rows_num,embedding_dim)*(embedding_dims,product_rows_num,embedding_dim)
            # ->(batch_size,product_rows_num,embedding_dim)*相同形状->相同形状
            kp=tf.multiply(tf.expand_dims(X_p,1),self.kernel)
            kp = tf.transpose(tf.reduce_sum(kp, axis=-1),perm=[0,2,1])
            result=tf.reduce_sum(tf.multiply(kp,X_q,),axis=-1)
        return result


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


class ONNLayer(tf.keras.layers.Layer):
    """
    input = {'item_tag1': tf.constant([0, 1, 2, 3]), 'item_tag2': tf.constant([4, 5, 6, 7]),
             'item_tag3': tf.constant([8, 9, 10, 11]),'user_tag0': tf.constant([12, 13, 14, 15])}
    input_dic = {}
    layer = ONNLayer(embedding_dims=8)
    for feature in layer.feature_names:
        input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    output = layer(input_dic)
    model = tf.keras.Model(input_dic, output)
    model.summary()
    print(model(input))
    """
    def __init__(self, feature_names=['item_tag1', 'item_tag2', 'item_tag3','user_tag0'],feature_dims=20, embedding_dims=16,mlp_units=[40,20],reduce=False, **kwargs):
        super(ONNLayer, self).__init__(**kwargs)
        self.feature_names=feature_names
        self.feature_dims=feature_dims
        self.fields_cnt=len(self.feature_names)
        self.embedding_dims = embedding_dims

        self.embedding_single=tf.keras.layers.Embedding(self.feature_dims,
                                                        self.embedding_dims,
                                                        embeddings_regularizer="l2")
        self.embedding_pair_dict = {}
        # 这种写法适合每个特征的各种取值从0开始独立编码，否则使用公共编码Embedding的参数使用率很低
        for i in range(self.fields_cnt):
            for j in range(i +1, self.fields_cnt):
                self.embedding_pair_dict[(i,j)]=(tf.keras.layers.Embedding(self.feature_dims,
                                                                           self.embedding_dims,
                                                                           embeddings_regularizer="l2"),
                                                 tf.keras.layers.Embedding(self.feature_dims,
                                                                           self.embedding_dims,
                                                                           embeddings_regularizer="l2")
                                                 )

        self.mlp_layer=make_mlp_layer(mlp_units,sigmoid_units=True)

        self.reduce=reduce


    def call(self, inputs):
        X=[]
        for feature in self.feature_names:
            X.append(inputs[feature])
        X=tf.concat(X,axis=1)
        X_single=tf.keras.layers.Flatten()(self.embedding_single(X))

        X_pair=[]
        for i in range(self.fields_cnt):
            for j in range(i+1,self.fields_cnt):
                embedding1,embedding2=self.embedding_pair_dict[(i,j)]
                feature1,feature2=inputs[self.feature_names[i]],inputs[self.feature_names[j]]
                X_pair.append(tf.multiply(embedding1(feature1),embedding2(feature2)))
        X_pair=tf.concat(X_pair,axis=1)
        if self.reduce:
            X_pair=tf.reduce_sum(X_pair,axis=2)
        X_pair = tf.keras.layers.Flatten()(X_pair)

        X_combined=tf.concat([X_single,X_pair],axis=1)

        output=self.mlp_layer(X_combined)

        result={'output':output}
        return result


class ParralledOnnLayer(tf.keras.layers.Layer):
    """
    不使用for循环
    input = {'item_tag1': tf.constant([0, 1, 2, 3]), 'item_tag2': tf.constant([4, 5, 6, 7]),
             'item_tag3': tf.constant([8, 9, 10, 11]),'user_tag0': tf.constant([12, 13, 14, 15])}
    input_dic = {}
    layer = ParralledOnnLayer(embedding_dims=8)
    for feature in layer.feature_names:
        input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    output = layer(input_dic)
    model = tf.keras.Model(input_dic, output)
    model.summary()
    print(model(input))
    """
    def __init__(self, feature_names=['item_tag1', 'item_tag2', 'item_tag3','user_tag0'],feature_dims=20, embedding_dims=16,mlp_units=[40,20],reduce=False, **kwargs):
        super(ParralledOnnLayer, self).__init__(**kwargs)
        self.feature_names=feature_names
        self.feature_dims=feature_dims
        self.fields_cnt=len(self.feature_names)
        self.embedding_dims = embedding_dims

        self.embedding_single=tf.keras.layers.Embedding(self.feature_dims,
                                                        self.embedding_dims,
                                                        embeddings_regularizer="l2")

        self.fa_interaction_layer = FieldAwareInteractionLayer(self.fields_cnt)

        self.mlp_layer=make_mlp_layer(mlp_units,sigmoid_units=True)

        self.reduce=reduce


    def call(self, inputs):
        X=[]
        for feature in self.feature_names:
            X.append(inputs[feature])
        X=tf.concat(X,axis=1)
        X_single=tf.keras.layers.Flatten()(self.embedding_single(X))

        X_pair=self.fa_interaction_layer(X)
        if self.reduce:
            X_pair=tf.reduce_sum(X_pair,axis=2)
        X_pair = tf.keras.layers.Flatten()(X_pair)

        X_combined=tf.concat([X_single,X_pair],axis=1)

        output=self.mlp_layer(X_combined)

        result={'output':output}
        return result



if __name__ == '__main__':
    """
    input = {'item_tag1': tf.constant([0, 1, 2, 3]), 'item_tag2': tf.constant([4, 5, 6, 7]),
             'item_tag3': tf.constant([8, 9, 10, 11]),'user_tag0': tf.constant([12, 13, 14, 15]),
             'user_tag1': tf.constant([16, 17, 18, 19])}
    input_dic = {}
    layer = PNNLayer(embedding_dims=8,method='outer',kernel_type='num')
    print(layer(input))
    for feature in layer.feature_names:
        input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    output = layer(input_dic)
    model = tf.keras.Model(input_dic, output)
    model.summary()
    print(model(input))
    """
    """
    input = {'item_tag1': tf.constant([0, 1, 2, 3]), 'item_tag2': tf.constant([4, 5, 6, 7]),
             'item_tag3': tf.constant([8, 9, 10, 11]), 'user_tag0': tf.constant([12, 13, 14, 15])}
    input_dic = {}
    layer = ParralledOnnLayer(embedding_dims=8)
    for feature in layer.feature_names:
        input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    output = layer(input_dic)
    model = tf.keras.Model(input_dic, output)
    model.summary()
    print(model(input))
    """
    input = {'user_tag0': tf.constant([12, 13, 14, 15]), 'user_tag1': tf.constant([16, 17, 18, 19]),
             'item_tag1': tf.constant([0, 1, 2, 3]), 'item_tag2': tf.constant([4, 5, 6, 7]),
             'item_tag3': tf.constant([8, 9, 10, 11])}
    input_dic = {}
    layer = DeepFMRankingLayer(embedding_dims=8)
    for feature in layer.feature_names:
        input_dic[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    output = layer(input_dic)
    model = tf.keras.Model(input_dic, output)
    model.summary()
    print(model(input))
