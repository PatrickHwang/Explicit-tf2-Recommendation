import itertools

import tensorflow as tf
from tensorflow.python.ops.init_ops_v2 import glorot_normal
from tensorflow.python.ops.math_ops import MatMul
from tensorflow.python.ops.nn_ops import BiasAdd
from tensorflow.python.keras import activations, initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import Input, Dense,Dropout,Conv2D,MaxPool2D
from tensorflow.keras.layers import BatchNormalization

import numpy as np

assert tf.__version__[0] == '2', 'model is built by tensorflow2,please check tf version first.'

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

class FiBiNetLayer(tf.keras.layers.Layer):
    """
        mm = ModelManager(layer='FiBiNet',allow_continuous=True,model_params={'cin_size':[16,32,64]})
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
                 embedding_dims=16, units=[128, 16], activation='relu',bilinear_type='interaction',reduction_ratio=3):
        super(FiBiNetLayer, self).__init__()
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.bilinear_type=bilinear_type
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims, embedding_dims)
        self.dnn_layer=MLPLayer(units,activation=activation)
        self.output_layer = Dense(1, activation='sigmoid')
        self.SENet = SENetLayer(reduction_ratio=reduction_ratio)
        self.Bilinear=BilinearInteractionLayer(bilinear_type=bilinear_type)

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


        X_emb = self.embedding_layer(X_cate)
        senet_output=self.SENet(X_emb)

        raw_bilinear_out=self.Bilinear(X_emb)
        senet_bilinear_out=self.Bilinear(senet_output)

        dnn_input=tf.keras.layers.Flatten()(tf.concat([raw_bilinear_out,senet_bilinear_out],axis=1))
        dnn_input=tf.concat([dnn_input,X_cont],axis=1)
        dnn_output=self.dnn_layer(dnn_input)
        # 没有完全加入wide的部分，也可以考虑加入
        output=self.output_layer(dnn_output)


        result = {'output': output}
        return result

class SENetLayer(tf.keras.layers.Layer):
    def __init__(self,reduction_ratio):
        super(SENetLayer,self).__init__()
        self.reduction_ratio=reduction_ratio

    def build(self,input_shape):
        self.field_num=input_shape[1]
        self.embedding_size=input_shape[2]

        self.mid_unit_num=max(1,self.field_num//self.reduction_ratio)
        self.excitation=MLPLayer([self.mid_unit_num,self.field_num],activation='relu', use_bias=False)
        super(SENetLayer, self).build(input_shape)

    def call(self,inputs):
        # (None,field_num,embedding_size) -> (None,field_num)
        Z=tf.reduce_mean(inputs,axis=-1)
        # (None,field_num) -> (None,mid_unit_num) -> (None,field_num)
        A=self.excitation(Z)
        # (None,field_num,embedding_size)*(None,field_num,1)
        V=tf.multiply(inputs,tf.expand_dims(A,axis=2))
        return V

class BilinearInteractionLayer(tf.keras.layers.Layer):
    def __init__(self,bilinear_type='interaction'):
        super(BilinearInteractionLayer,self).__init__()
        self.bilinear_type=bilinear_type

    def build(self,input_shape):
        self.field_num=input_shape[1]
        self.embedding_size=input_shape[2]
        if self.bilinear_type=='all':
            self.W=self.add_weight(shape=(self.embedding_size,self.embedding_size),initializer=glorot_normal(), name="bilinear_weight")
        elif self.bilinear_type=='each':
            self.W_list=[self.add_weight(shape=(self.embedding_size,self.embedding_size),initializer=glorot_normal(), name="bilinear_weight"+str(i)) for i in range(self.field_num-1)]
        elif self.bilinear_type=='interaction':
            self.W_list=[self.add_weight(shape=(self.embedding_size,self.embedding_size),initializer=glorot_normal(), name="bilinear_weight"+str(i)+'_'+str(j)) for i,j in itertools.combinations(range(self.field_num), 2)]
        else:
            raise NotImplementedError
        super(BilinearInteractionLayer, self).build(input_shape)


    def call(self,inputs):
        # (None,field_num,embedding_size) -> [n*(None,embedding_size)]
        field_list=tf.split(inputs,self.field_num,axis=1)
        if self.bilinear_type=='all':
            p=[tf.multiply(tf.tensordot(field_list[i],self.W,axes=(-1,0)),field_list[j]) for i,j in itertools.combinations(range(self.field_num), 2)]
        elif self.bilinear_type=='each':
            p=[tf.multiply(tf.tensordot(field_list[i],self.W_list[i],axes=(-1,0)),field_list[j]) for i,j in itertools.combinations(range(self.field_num), 2)]
        elif self.bilinear_type=='interaction':
            p=[tf.multiply(tf.tensordot(field_list[v[0]],w,axes=(-1,0)),field_list[v[1]]) for v,w in zip(itertools.combinations(range(self.field_num), 2),self.W_list)]
        else:
            raise NotImplementedError
        # (None,fields_interaction_num,embedding_size)
        output=tf.concat(p,axis=1)

        return output

class TransformerAttentionLayer(tf.keras.layers.Layer):
    def __init__(self,num_heads=2,use_res=True,res_learnable=False,scaling=False):
        super(TransformerAttentionLayer,self).__init__()
        self.num_heads=num_heads
        self.use_res=use_res
        self.scaling=scaling
        self.res_learnable=res_learnable

    def build(self,input_shape):
        super(TransformerAttentionLayer, self).build(input_shape)
        self.embedding_dim=input_shape[-1]
        self.att_embedding_size=self.embedding_dim//self.num_heads
        assert self.embedding_dim%self.num_heads==0,'请不要整活,不能整除的多头注意力不方便加残差'
        self.W_query=self.add_weight(name='query',shape=[self.embedding_dim,self.embedding_dim],dtype=tf.float32,
                                     initializer=tf.keras.initializers.TruncatedNormal())
        self.W_key = self.add_weight(name='key', shape=[self.embedding_dim, self.embedding_dim], dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal())
        self.W_value = self.add_weight(name='value', shape=[self.embedding_dim, self.embedding_dim], dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal())
        if self.use_res and self.res_learnable:
            self.W_res=self.add_weight(name='res', shape=[self.embedding_dim, self.embedding_dim], dtype=tf.float32,
                                       initializer=tf.keras.initializers.TruncatedNormal())


    def call(self,inputs):
        # (None,field_num,embedding_size) -> 不变
        querys=tf.tensordot(inputs,self.W_query,axes=(-1,0))
        keys = tf.tensordot(inputs, self.W_key, axes=(-1, 0))
        values = tf.tensordot(inputs, self.W_value, axes=(-1, 0))

        # 把embedding维分头取到第一维中
        # (num_heads,None,field_num,att_embedding_size)
        querys=tf.stack(tf.split(querys,self.num_heads,axis=-1))
        keys = tf.stack(tf.split(keys, self.num_heads, axis=-1))
        values = tf.stack(tf.split(values, self.num_heads, axis=-1))

        # 点积注意力，Q和K前面的维不变，Q矩阵乘法K的转置
        # (num_heads,None,field_num,field_num)
        inner_product=tf.matmul(querys,keys,transpose_b=True)
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        attention_scores=tf.nn.softmax(inner_product,axis=1)

        # (num_heads,None,field_num,att_embedding_size)
        result=tf.matmul(attention_scores,values)
        # 还原heads到最后一维拼回原embedding_size
        # (1,None,field_num,embedding_size) -> (None,field_num,embedding_size)
        result=tf.concat(tf.split(result,self.num_heads),axis=-1)
        result=tf.squeeze(result,axis=0)

        if self.use_res and self.res_learnable:
            result+=tf.tensordot(inputs,self.W_res,axes=(-1,0))
        elif self.use_res and not self.res_learnable:
            result+=inputs

        result=tf.nn.relu(result)
        return result

class AutoIntLayer(tf.keras.layers.Layer):
    """
        mm = ModelManager(layer='AutoInt',allow_continuous=True,model_params={'cin_size':[16,32,64]})
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
                 embedding_dims=8, units=[128, 16], activation='relu',attention_layer_num=2,num_heads=2):
        super(AutoIntLayer, self).__init__()
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims, embedding_dims)
        self.continuous_embedding=tf.keras.layers.Embedding(len(continuous_features),embedding_dims)
        self.attention_layers=[TransformerAttentionLayer(num_heads=num_heads) for _ in range(attention_layer_num)]
        self.dnn_layer = MLPLayer(units, activation=activation)
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


        X_cate_emb = self.embedding_layer(X_cate)

        # 按论文的特征处理方式，每个连续变量学习一个embedding并乘上归一化后的连续值（此处归一化省略），然后拼接离散embedding
        X_cont_emb=self.continuous_embedding(tf.reshape(tf.range(len(self.continuous_features)),(1,-1)))
        X_cont_emb=tf.multiply(X_cont_emb,tf.expand_dims(X_cont,axis=-1))

        X_emb=tf.concat([X_cate_emb,X_cont_emb],axis=1)

        att_input=X_emb
        for layer in self.attention_layers:
            att_input=layer(att_input)

        dnn_input=tf.keras.layers.Flatten()(att_input)


        dnn_output=self.dnn_layer(dnn_input)
        output=self.output_layer(dnn_output)


        result = {'output': output}
        return result

if __name__ == '__main__':
    input = tf.random.normal((2, 6, 8))
    layer = TransformerAttentionLayer(2,True,True)
    print(layer(input).shape)