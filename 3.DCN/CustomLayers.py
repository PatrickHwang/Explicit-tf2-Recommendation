import tensorflow as tf
from tensorflow.python.ops.math_ops import MatMul
from tensorflow.python.ops.nn_ops import BiasAdd
from tensorflow.python.keras import activations, initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import Input, Dense,Dropout,Conv2D,MaxPool2D
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
        print(X)

        w_output = self.w(X)
        emb_output = self.embed(X)
        print(emb_output.shape)

        first_order = tf.reduce_sum(w_output, axis=1)

        sum_of_square = tf.reduce_sum(tf.square(emb_output), axis=1)
        square_of_sum = tf.square(tf.reduce_sum(emb_output, axis=1))
        second_order = 0.5 * tf.reduce_sum(tf.subtract(square_of_sum, sum_of_square), axis=1, keepdims=True)

        output = tf.nn.sigmoid(self.bias + first_order + second_order)
        result = {'output': output}
        return result


class WideAndDeepRankingLayer(tf.keras.layers.Layer):
    """
    mm = ModelManager(layer='wide_and_deep')
    input = {'user_tag1':tf.constant([12,13,14,15]),'user_tag2':tf.constant([16,17,18,19]),
             'item_tag1': tf.constant([0, 1, 2,3]), 'item_tag2': tf.constant([4, 5, 6,7]),
             'item_tag3': tf.constant([8,9,10,11])}
    print(mm.model(input))
    """

    def __init__(self,
                 categorical_features=['uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'itag1', 'itag2', 'itag3',
                                       'itag4'],
                 continuous_features=['itag4_origin', 'itag4_square', 'itag4_cube'], feature_dims=160000,
                 embedding_dims=16,
                 mlp_dims=[32, 8], **kwargs):
        super(WideAndDeepRankingLayer, self).__init__(**kwargs)
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.feature_dims = feature_dims
        self.embedding_dims = embedding_dims
        self.mlp_dims = mlp_dims

    def build(self, input_shape):
        # embedding and w
        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims,
                                               embeddings_regularizer="l2")

        self.MLP_layer1 = MLPLayer(units=self.mlp_dims, activation='relu')
        self.MLP_layer2 = MLPLayer(units=[1], activation='sigmoid')
        self.built = True

    def call(self, inputs):
        # 需要有labelencode以前的原始特征值才行，dataset需要另外处理
        X = []
        for feature in self.categorical_features:
            X.append(inputs[feature])
        X = tf.concat(X, axis=1)

        cont = []
        for feature in self.continuous_features:
            cont.append(inputs[feature])
        cont = tf.concat(cont, axis=1)

        # common input vectors
        emb_output = self.embed(X)

        # DNN Part
        dense_embedding = tf.keras.layers.Flatten()(emb_output)
        dnn_vector = self.MLP_layer1(dense_embedding)
        # dnn_vector=tf.cast(dnn_vector,dtype=float)

        # output
        combined_vector = tf.concat([cont, dnn_vector], axis=1)
        output = self.MLP_layer2(combined_vector)

        result = {'output': output}
        return result


class DenseLayer(tf.keras.layers.Layer):
    """使用Dense直接实现MLP"""

    def __init__(self, units, activation):
        super(DenseLayer, self).__init__()
        self.hidden_layer = [Dense(x, activation=activation) for x in units]

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        return x


class CrossLayer(tf.keras.layers.Layer):
    def __init__(self, layer_num, reg_w=1e-4, reg_b=1e-4):
        super(CrossLayer, self).__init__()
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        self.cross_weight = [
            self.add_weight(name='w' + str(i),
                            shape=(input_shape[1], 1),
                            initializer=tf.random_normal_initializer(),
                            regularizer=tf.keras.regularizers.l2(self.reg_w),
                            trainable=True)
            for i in range(self.layer_num)
        ]
        self.cross_bias = [
            self.add_weight(name='b' + str(i),
                            shape=(input_shape[1], 1),
                            initializer=tf.zeros_initializer(),
                            regularizer=tf.keras.regularizers.l2(self.reg_b),
                            trainable=True)
            for i in range(self.layer_num)
        ]

    def call(self, inputs, **kwargs):
        x0 = tf.expand_dims(inputs, axis=2)  # (None,dims,1)
        xl = x0  # (None,dims,1)
        for i in range(self.layer_num):
            xl_w = tf.matmul(tf.transpose(xl, [0, 2, 1]), self.cross_weight[i])  # (None,1,dim) * (dim,1) -> (None,1,1)]
            xl = tf.matmul(x0, xl_w) + self.cross_bias[
                i] + xl  # (None,dims,1) * (None,1,1)+(dims,1)+(None,dims,1) -> (None,dims,1)
        output = tf.squeeze(xl, axis=2)  # (None,dims)
        return output


class DeepCrossNetworkLayer(tf.keras.layers.Layer):
    """
    mm = ModelManager(layer='dcn_ranking',allow_continuous=True,model_params={'type':'matrix'})
    input = {'uid': tf.constant([0, 1, 2]), 'iid': tf.constant([3, 4, 5]),
             'utag1': tf.constant([6, 7, 8]), 'utag2': tf.constant([9, 10, 11]),
             'utag3': tf.constant([12, 13, 14]), 'utag4': tf.constant([15, 16, 17]),
             'itag1': tf.constant([18, 19, 20]), 'itag2': tf.constant([21, 22, 23]),
             'itag3': tf.constant([24, 25, 26]), 'itag4': tf.constant([27, 28, 29]),
             'itag4_origin':tf.constant([0.2, 7.8, 4.9]),
             'itag4_square':tf.constant([5.3, 1.2, 8.0]),
             'itag4_cube': tf.constant([-3.8, -19.6, 4.2])}
    print(mm.model(input))
    """

    def __init__(self,
                 categorical_features=['uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'itag1', 'itag2', 'itag3',
                                       'itag4'],
                 continuous_features=['itag4_origin', 'itag4_square', 'itag4_cube'], feature_dims=160000,
                 embedding_dims=16, units=[64, 8], activation='relu', layer_num=3, reg_w=1e-4, reg_b=1e-4, type='vec'):
        super(DeepCrossNetworkLayer, self).__init__()
        if type == 'vec':
            self.cross_layer = CrossLayer(layer_num, reg_w, reg_b)
        else:
            self.cross_layer = MatrixCrossLayer(layer_num, reg_w, reg_b)
        self.dense_layer = DenseLayer(units, activation)
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims, embedding_dims)
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.output_layer = Dense(1, activation='sigmoid')

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        X = []
        for feature in self.categorical_features:
            X.append(inputs[feature])
        X = tf.concat(X, axis=1)

        cont = []
        for feature in self.continuous_features:
            cont.append(inputs[feature])
        X_cont = tf.concat(cont, axis=1)

        X_emb = self.embedding_layer(X)
        X_flatten = tf.keras.layers.Flatten()(X_emb)

        _input = tf.concat([X_cont, X_flatten], axis=1)

        cross_output = self.cross_layer(_input)

        dnn_output = self.dense_layer(_input)

        combine_output = tf.concat([cross_output, dnn_output], axis=1)

        output = self.output_layer(combine_output)
        result = {'output': output}
        return result


class MatrixCrossLayer(tf.keras.layers.Layer):
    def __init__(self, layer_num, reg_w=1e-4, reg_b=1e-4):
        super(MatrixCrossLayer, self).__init__()
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        self.cross_weight = [
            self.add_weight(name='w' + str(i),
                            shape=(input_shape[1], input_shape[1]),
                            initializer=tf.random_normal_initializer(),
                            regularizer=tf.keras.regularizers.l2(self.reg_w),
                            trainable=True)
            for i in range(self.layer_num)
        ]
        self.cross_bias = [
            self.add_weight(name='b' + str(i),
                            shape=(input_shape[1], 1),
                            initializer=tf.zeros_initializer(),
                            regularizer=tf.keras.regularizers.l2(self.reg_b),
                            trainable=True)
            for i in range(self.layer_num)
        ]

    def call(self, inputs, **kwargs):
        x0 = tf.expand_dims(inputs, axis=2)  # (None,dims,1)
        xl = x0  # (None,dims,1)
        for i in range(self.layer_num):
            xl_w = tf.matmul(self.cross_weight[i], xl)  # (None,dims,dims) * (dims,1) -> (None,dims,1)]
            xl = tf.multiply(x0, xl_w + self.cross_bias[
                i]) + xl  # (None,dims,1) dot ((None,dims,1)+(dims,1))+(None,dims,1) -> (None,dims,1)
        output = tf.squeeze(xl, axis=2)  # (None,dims)
        return output


class XDeepFMRankingLayer(tf.keras.layers.Layer):
    """
        mm = ModelManager(layer='xDeepFM',allow_continuous=True,model_params={'cin_size':[16,32,64]})
        input = {'uid': tf.constant([0, 1, 2]), 'iid': tf.constant([3, 4, 5]),
                 'utag1': tf.constant([6, 7, 8]), 'utag2': tf.constant([9, 10, 11]),
                 'utag3': tf.constant([12, 13, 14]), 'utag4': tf.constant([15, 16, 17]),
                 'itag1': tf.constant([18, 19, 20]), 'itag2': tf.constant([21, 22, 23]),
                 'itag3': tf.constant([24, 25, 26]), 'itag4': tf.constant([27, 28, 29]),
                 'itag4_origin':tf.constant([0.2, 7.8, 4.9]),
                 'itag4_square':tf.constant([5.3, 1.2, 8.0]),
                 'itag4_cube': tf.constant([-3.8, -19.6, 4.2])}
        print(mm.model(input))
        """

    def __init__(self,
                 categorical_features=['uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'itag1', 'itag2', 'itag3',
                                       'itag4'],
                 continuous_features=['itag4_origin', 'itag4_square', 'itag4_cube'], feature_dims=160000,
                 embedding_dims=16, units=[64, 8], activation='relu', cin_size=[16, 32, 64]):
        super(XDeepFMRankingLayer, self).__init__()
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.w = tf.keras.layers.Embedding(feature_dims,
                                           1,
                                           embeddings_regularizer="l2")
        self.dense_layer = DenseLayer(units, activation)
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims, embedding_dims)
        self.cin_layer = CINLayer(cin_size)
        self.output_layer = Dense(1, activation='sigmoid')

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        cate = []
        for feature in self.categorical_features:
            cate.append(inputs[feature])
        X_cate = tf.concat(cate, axis=1)

        cont = []
        for feature in self.continuous_features:
            cont.append(inputs[feature])
        X_cont = tf.concat(cont, axis=1)

        # linear_part
        w_output = self.w(X_cate)
        linear_part = tf.reduce_sum(w_output, axis=1)

        # dense_part
        X_emb = self.embedding_layer(X_cate)
        X_flatten = tf.keras.layers.Flatten()(X_emb)
        dense_input = tf.concat([X_cont, X_flatten], axis=1)
        dense_part = self.dense_layer(dense_input)

        # cin_part
        cin_part = self.cin_layer(X_emb)

        # output
        output = self.output_layer(tf.concat([linear_part, dense_part, cin_part], axis=1))

        result = {'output': output}
        return result


class CINLayer(tf.keras.layers.Layer):
    """
    input = tf.constant(np.arange(24).reshape(2, 3, 4), dtype=float)
    cin_layer = CINLayer([2, 4])
    print(cin_layer(input))
    """

    def __init__(self, cin_size=[8, 16]):
        super(CINLayer, self).__init__()
        self.cin_size = cin_size

    def build(self, input_shape):
        self.field_num = [input_shape[1]] + self.cin_size

        self.cin_W = [self.add_weight(
            name='w' + str(i),
            shape=(1, self.field_num[0] * self.field_num[i], self.field_num[i + 1]),
            initializer=tf.initializers.glorot_uniform(),
            regularizer=tf.keras.regularizers.l1_l2(1e-5),
            trainable=True)
            for i in range(len(self.field_num) - 1)]

    def call(self, inputs):
        embedding_dim = inputs.shape[-1]
        res_list = [inputs]
        X0 = tf.split(inputs, embedding_dim, axis=-1)  # list: embedding_dim * [None, field_num[0], 1]
        for i, size in enumerate(self.field_num[1:]):
            Xi = tf.split(res_list[-1], embedding_dim, axis=-1)  # list: embedding_dim * [None, field_num[i], 1]
            x = tf.matmul(X0, Xi, transpose_b=True)  # list: embedding_dim * [None, field_num[0], field_num[i]]
            x = tf.reshape(x, shape=[embedding_dim, -1, self.field_num[0] * self.field_num[
                i]])  # [embedding_dim,None, field_num[0]*field_num[i]]
            x = tf.transpose(x, [1, 0, 2])  # [None,embedding_dim,self.field_num[0]*self.field_num[i]]
            x = tf.nn.conv1d(input=x, filters=self.cin_W[i], stride=1,
                             padding='VALID')  # filter:(1, self.field_num[0]*self.field_num[i], self.field_num[i+1])
            # (None, embedding_dim, field_num[i+1])
            x = tf.transpose(x, [0, 2, 1])  # (None,field_num[i+1],embedding_dim)
            res_list.append(x)
        res_list = res_list[1:]  # 去掉原始输入
        res = tf.concat(res_list, axis=1)  # (None,sum(self.field_num[1:]),embedding_dim)
        output = tf.reduce_sum(res, axis=-1)  # (None,sum(self.field_num[1:]),-1)
        return output


class NeuralFactorizationMachineLayer(tf.keras.layers.Layer):
    def __init__(self, categorical_features=['uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'itag1', 'itag2', 'itag3','itag4'],
                continuous_features=['itag4_origin', 'itag4_square', 'itag4_cube'], feature_dims=160000,embedding_dims=16,
                units=[64, 8], activation='relu'):
        super(NeuralFactorizationMachineLayer, self).__init__()
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.feature_dims=feature_dims
        self.embedding_dims=embedding_dims
        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims,
                                               embeddings_regularizer="l2")
        self.MLP_layer1 = MLPLayer(units=units, activation=activation)
        self.MLP_layer2 = MLPLayer(units=[1], activation='sigmoid')
        self.bn_layer = BatchNormalization()

    def build(self, input_shape):
        # embedding and w
        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims,
                                               embeddings_regularizer="l2")
        self.w = tf.keras.layers.Embedding(self.feature_dims,
                                           1,
                                           embeddings_regularizer="l2")
        self.built = True

    def call(self, inputs):
        X_cont=[]
        X_cate = []
        for feature in self.continuous_features:
            X_cont.append(inputs[feature])
        X_cont = tf.concat(X_cont, axis=1)

        for feature in self.categorical_features:
            X_cate.append(inputs[feature])
        X_emb = tf.concat(X_cate, axis=1)

        # w_output = self.w(X_cate)     暂时不使用一阶特征
        emb_output = self.embed(X_emb)

        # first_order = tf.reduce_sum(w_output, axis=1)

        sum_of_square = tf.reduce_sum(tf.square(emb_output), axis=1)
        square_of_sum = tf.square(tf.reduce_sum(emb_output, axis=1))
        second_order = 0.5*tf.subtract(square_of_sum, sum_of_square)

        combined_vector=tf.concat([second_order,X_cont],axis=1)
        combined_vector=self.bn_layer(combined_vector)
        output = self.MLP_layer1(combined_vector)
        output=self.MLP_layer2(output)

        result = {'output': output}
        return result

class DeepCrossingLayer(tf.keras.layers.Layer):
    def __init__(self,categorical_features=['uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'itag1', 'itag2', 'itag3','itag4'],
                continuous_features=['itag4_origin', 'itag4_square', 'itag4_cube'], feature_dims=160000,embedding_dims=16,
                units=[32],activation='relu',res_layer_num=2):
        super(DeepCrossingLayer, self).__init__()
        self.categorical_features=categorical_features
        self.continuous_features=continuous_features
        self.feature_dims=feature_dims
        self.embedding_dims=embedding_dims
        self.units=units
        self.activation=activation
        self.res_layer_num=res_layer_num
        self.output_layer = Dense(1, activation='sigmoid')

    def build(self,input_shape):
        self.embed = tf.keras.layers.Embedding(self.feature_dims,
                                               self.embedding_dims,
                                               embeddings_regularizer="l2")

        self.res_layers=[ResLayer(self.units,self.activation) for _ in range(self.res_layer_num)]

    def call(self,inputs):
        X_cont = []
        X_cate = []
        for feature in self.continuous_features:
            X_cont.append(inputs[feature])
        X_cont = tf.concat(X_cont, axis=1)

        for feature in self.categorical_features:
            X_cate.append(inputs[feature])
        X_emb = tf.concat(X_cate, axis=1)

        emb_output = tf.keras.layers.Flatten()(self.embed(X_emb))
        X_combined=tf.concat([emb_output,X_cont],axis=1)
        for layer in self.res_layers:
            X_combined=layer(X_combined)
        output=self.output_layer(X_combined)

        result = {'output': output}
        return result




class ResLayer(tf.keras.layers.Layer):
    def __init__(self,hidden_units,activation):
        super(ResLayer,self).__init__()
        self.dense_layer=[Dense(i,activation=activation) for i in hidden_units]

    def build(self,input_shape):
        dim=input_shape[1]
        self.output_layer=Dense(dim,activation=None)

    def call(self,inputs):
        x=inputs
        for layer in self.dense_layer:
            x=layer(x)
        x=self.output_layer(x)
        output=x+inputs
        return tf.nn.relu(output)

class FNNLayer(tf.keras.layers.Layer):
    '''
    先预训练一个FM模型
    mm = ModelManager(layer='fm_ranking', allow_continuous=True,
                      checkpoint_dir='fm_model/checkpoint', tensorboard_dir='fm_model/tensorboard',
                      output_dir='fm_model/output')
    mm.run()
    '''
    def __init__(self,categorical_features=['uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'itag1', 'itag2', 'itag3','itag4'],feature_dims=150000,embedding_dims=16,units=[64,32,8],activation='relu',is_batch_norm=True):
        super(FNNLayer, self).__init__()
        self.categorical_features=categorical_features
        self.feature_dims=feature_dims
        self.embedding_dims=embedding_dims

        self.MLP_layer1 = MLPLayer(units=units, activation=activation,is_batch_norm=is_batch_norm)
        self.MLP_layer2 = MLPLayer(units=[1], activation='sigmoid')
        fm_model=tf.saved_model.load('fm_model/output')
        self.embedding_table = fm_model.variables[1]
        self.embedding_table=tf.constant(self.embedding_table)
        #self.embed=fm_model.embed
        #self.embed.trainable = False

    def build(self,input_shape):
        self.built=True

    def call(self,inputs):
        X_cate=[]
        for feature in self.categorical_features:
            X_cate.append(inputs[feature])
        lookup_indexes = tf.concat(X_cate, axis=1)
        X_emb=tf.nn.embedding_lookup(self.embedding_table,lookup_indexes)

        emb_output = tf.keras.layers.Flatten()(X_emb)
        output=self.MLP_layer1(emb_output)
        output=self.MLP_layer2(output)

        result = {'output': output}
        return result

class KMaxPool(tf.keras.layers.Layer):
    def __init__(self,k):
        super(KMaxPool,self).__init__()
        self.k=k

    def build(self,input_shape):
        self.built=True

    def call(self,inputs):
        # 要求四维输入 (None,fields_num,ebd_dim,1)
        # (None,1,ebd_dim,fields_num)
        inputs=tf.transpose(inputs,[0,3,2,1])
        # 对最后一维取k个最大值
        k_max=tf.nn.top_k(inputs,k=self.k,sorted=True)[0]
        # 恢复(None,fields_num,ebd_dim,1)
        output=tf.transpose(k_max,[0,3,2,1])
        return output

class CCPMBaseLayer(tf.keras.layers.Layer):
    """
    input = tf.constant(tf.random.normal((6,13,8)), dtype=float)
    layer = CCPMLayer()
    print(layer(input))
    """
    def __init__(self,filters=[4,6],kernel_width=[4,2]):
        # filters:输出通道数,kernel_width:卷积核长度
        super(CCPMBaseLayer,self).__init__()
        self.filters=filters
        self.kernel_width=kernel_width
        self.conv_layers=[]
        self.kmax_layers=[]
        self.flatten_layer=tf.keras.layers.Flatten()
        self.layers_num = len(self.filters)

    def build(self,input_shape):
        fields_num=input_shape[-1]
        for i in range(self.layers_num):
            self.conv_layers.append(Conv2D(filters=self.filters[i],
                                    kernel_size=(self.kernel_width[i],1),
                                    strides=(1,1),
                                    padding='same',
                                    activation='tanh'))
            # 从1开始计数的层序号
            j=i+1
            k=max(1,int((1-pow(j/self.layers_num,self.layers_num-j))*fields_num)) if j<self.layers_num else 3
            self.kmax_layers.append(KMaxPool(k=k))

    def call(self,inputs):
        # (None,fields_num,ebd_dim) -> (None,fields_num,ebd_dim,1)
        # 增加的是输出通道维
        x=tf.expand_dims(inputs,axis=-1)
        for i in range(self.layers_num):
            x=self.conv_layers[i](x)
            x=self.kmax_layers[i](x)
        output=self.flatten_layer(x)
        return output

class CCPMLayer(tf.keras.layers.Layer):
    def __init__(self,categorical_features=['uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'itag1', 'itag2', 'itag3','itag4'],continuous_features=['itag4_origin', 'itag4_square', 'itag4_cube'],
                 feature_dims=150000,embedding_dims=16,
                 units=[64,32,8],activation='relu',is_batch_norm=True,
                 filters=[4,6],kernel_width=[4,2]):
        # filters:输出通道数,kernel_width:卷积核长度
        super(CCPMLayer,self).__init__()
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.units = units
        self.activation = activation
        self.MLP_layer1 = MLPLayer(units=units, activation=activation, is_batch_norm=is_batch_norm)
        self.MLP_layer2 = MLPLayer(units=[1], activation='sigmoid')
        self.ccpm_layer = CCPMBaseLayer(filters, kernel_width)
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims,
                                               embedding_dims,
                                               embeddings_regularizer="l2")

    def build(self,input_shape):
        self.built=True

    def call(self,inputs):
        X_cont = []
        X_cate = []
        for feature in self.continuous_features:
            X_cont.append(inputs[feature])
        X_cont = tf.concat(X_cont, axis=1)

        for feature in self.categorical_features:
            X_cate.append(inputs[feature])
        X_emb = tf.concat(X_cate, axis=1)
        X_emb=self.embedding_layer(X_emb)

        ccpm_output=self.ccpm_layer(X_emb)
        input=tf.concat([ccpm_output,X_cont],axis=1)
        output=self.MLP_layer1(input)
        output=self.MLP_layer2(output)

        result = {'output': output}
        return result

class FGCNNBaseLayer(tf.keras.layers.Layer):
    def __init__(self,filters=[14,16],kernel_width=[7,7],dnn_maps=[3,3],pooling_width=[2,2]):
        super(FGCNNBaseLayer,self).__init__()
        self.filters=filters
        self.kernel_width=kernel_width
        self.dnn_maps=dnn_maps
        self.pooling_width=pooling_width

    def build(self,input_shape):
        # input_shape:(None,n,k)
        self.conv_layers=[]
        self.pool_layers=[]
        self.dense_layers=[]
        for layer_index in range(len(self.filters)):
            self.conv_layers.append(
                Conv2D(filters=self.filters[layer_index],
                       kernel_size=(self.kernel_width[layer_index],1),
                       strides=(1,1),
                       padding='same',
                       activation='tanh')
            )
            self.pool_layers.append(
                MaxPool2D(pool_size=(self.pooling_width[layer_index],1))
            )
            self.dense_layers.append(
                Dense(self.dnn_maps[layer_index]*input_shape[1]*input_shape[2]//self.pooling_width[layer_index])
            )
        self.flatten_layer=tf.keras.layers.Flatten()

    def call(self,inputs):
        # inputs:(None,fields_cnt,emb_dims)
        emb_dims=inputs.shape[-1]
        dnn_outputs=[]
        # 新增频道维度
        # (None,fields_cnt,emb_dims) -> (None,fields_cnt,emb_dims,1)
        x=tf.expand_dims(inputs,axis=-1)
        for layer_index in range(len(self.filters)):
            x=self.conv_layers[layer_index](x)  # (None,fields_cnt,emb_dims,filters[layer_index]
            x=self.pool_layers[layer_index](x)  # (None,fields_cnt//pooling_width[layer_index],emb_dims,filters[layer_index])
            out=self.flatten_layer(x)           # (None,fields_cnt//pooling_width[layer_index]*emb_dims*filters[layer_index])
            out=self.dense_layers[layer_index](out)   # (None,self.dnn_maps[layer_index]*fields_cnt*emb_dims//self.pooling_width[layer_index])
            out = tf.reshape(out, shape=(-1, out.shape[1] // emb_dims, emb_dims)) # (None,self.dnn_maps[layer_index]*fields_cnt//self.pooling_width[layer_index],emb_dims)
            dnn_outputs.append(out)
        output=tf.concat(dnn_outputs,axis=1)
        return output

class FGCNNLayer(tf.keras.layers.Layer):
    def __init__(self,categorical_features=['uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'itag1', 'itag2', 'itag3','itag4'],continuous_features=['itag4_origin', 'itag4_square', 'itag4_cube'],
                 feature_dims=150000,embedding_dims=16,
                 units=[64,8],activation='relu',is_batch_norm=True,
                 filters=[14,16],kernel_width=[7,7],dnn_maps=[3,3],pooling_width=[2,2]):
        # filters:输出通道数,kernel_width:卷积核长度
        super(FGCNNLayer,self).__init__()
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.units = units
        self.activation = activation
        self.MLP_layer1 = MLPLayer(units=units, activation=activation, is_batch_norm=is_batch_norm)
        self.MLP_layer2 = MLPLayer(units=[1], activation='sigmoid')
        self.fgcnn_layer = FGCNNBaseLayer(filters=filters,kernel_width=kernel_width,dnn_maps=dnn_maps,pooling_width=pooling_width)
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims,
                                                         embedding_dims,
                                                         embeddings_regularizer="l2")

    def build(self,input_shape):
        self.built=True

    def call(self,inputs):
        X_cont = []
        X_cate = []
        for feature in self.continuous_features:
            X_cont.append(inputs[feature])
        X_cont = tf.concat(X_cont, axis=1)

        for feature in self.categorical_features:
            X_cate.append(inputs[feature])
        X_emb = tf.concat(X_cate, axis=1)
        X_emb = self.embedding_layer(X_emb)

        fgcnn_output = self.fgcnn_layer(X_emb)
        fgcnn_combine=tf.concat([X_emb,fgcnn_output],axis=1)
        fgcnn_combine=tf.keras.layers.Flatten()(fgcnn_combine)
        input = tf.concat([fgcnn_combine, X_cont], axis=1)
        output = self.MLP_layer1(input)
        output = self.MLP_layer2(output)

        result = {'output': output}
        return result

class InteractionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(InteractionLayer,self).__init__()

    def call(self,inputs):
        result=[]
        fields_cnt=inputs.shape[1]
        for i in range(fields_cnt-1):
            for j in range(i+1,fields_cnt):
                product=tf.multiply(inputs[:,i,:],inputs[:,j,:])
                result.append(product)
        result=tf.convert_to_tensor(result)
        result=tf.transpose(result,[1,0,2])
        return result

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self,attn_size):
        super(AttentionLayer, self).__init__()
        self.attention_w=Dense(attn_size,activation='relu')
        self.attention_h=Dense(1,activation=None)

    def call(self,inputs):                             # (None,fields,ebd_size)
        x=self.attention_h(self.attention_w(inputs))   # (None,fields,1)
        attention_score=tf.nn.softmax(x,axis=1)        # (None,fields,1)
        matmul_score=tf.transpose(attention_score,[0,2,1])  # (None,1,fields)
        output=tf.matmul(matmul_score,inputs)          # (None,1,ebd_size)
        output=tf.reshape(output,(-1,inputs.shape[2]))  # (None,ebd_size)
        return output

class AttentionalFactorizationMachine(tf.keras.layers.Layer):
    def __init__(self,
                 categorical_features=['uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'itag1', 'itag2', 'itag3','itag4'],
                 feature_dims=150000, embedding_dims=16,attn_size=3):
        # filters:输出通道数,kernel_width:卷积核长度
        super(AttentionalFactorizationMachine, self).__init__()
        self.categorical_features = categorical_features
        self.interaction_layer=InteractionLayer()
        self.attention_layer=AttentionLayer(attn_size=attn_size)
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims,
                                                         embedding_dims,
                                                         embeddings_regularizer="l2")
        self.output_layer = MLPLayer(units=[1], activation='sigmoid')

    def call(self,inputs):
        X = []
        for feature in self.categorical_features:
            X.append(inputs[feature])
        X = tf.concat(X, axis=1)
        emb_output = self.embedding_layer(X)

        interaction_output=self.interaction_layer(emb_output)
        attention_output=self.attention_layer(interaction_output)

        output=self.output_layer(attention_output)
        result = {'output': output}
        return result


if __name__ == '__main__':
    input = tf.random.normal((2,3,8))
    layer=AttentionLayer(3)
    print(layer(input).shape)

