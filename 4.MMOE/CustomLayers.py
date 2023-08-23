import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.math_ops import MatMul
from tensorflow.python.ops.nn_ops import BiasAdd
from tensorflow.python.keras import activations, initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import Input, Dense,Dropout,Conv2D,MaxPool2D
from tensorflow.keras.layers import BatchNormalization

from tensorflow.python.framework.errors_impl import InvalidArgumentError as wrapError

import numpy as np

assert tf.__version__[0] == '2', 'model is built by tensorflow2,please check tf version first.'

def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)

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

class MMOELayer(tf.keras.layers.Layer):
    """
        mm = ModelManager(layer='mmoe_layer',allow_continuous=True)
        input = {'sdk_type': tf.constant([0, 1, 2]), 'remote_host': tf.constant([3, 4, 5]),
                 'device_type': tf.constant([6, 7, 8]), 'dtu': tf.constant([9, 10, 11]),
                 'click_goods_num': tf.constant([12, 13, 14]), 'buy_click_num': tf.constant([15, 16, 17]),
                 'goods_show_num': tf.constant([18, 19, 20]), 'goods_click_num': tf.constant([21, 22, 23]),
                 'brand_name': tf.constant([24, 25, 26]),
                 'click_goods_num_origin':tf.constant([0.2, 7.8, 4.9]),
                 'click_goods_num_square':tf.constant([5.3, 1.2, 8.0]),
                 'click_goods_num_cube': tf.constant([-3.8, -19.6, 4.2])}
        print(mm.model(input))
        """
    def __init__(self,
                 categorical_features=['sdk_type', 'remote_host', 'device_type', 'dtu',
                                       'click_goods_num', 'buy_click_num', 'goods_show_num', 'goods_click_num',
                                       'brand_name'],
                 continuous_features=['click_goods_num_origin', 'click_goods_num_square', 'click_goods_num_cube'], feature_dims=160000,
                 embedding_dims=16, expert_num=3 ):
        super(MMOELayer, self).__init__()
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims, embedding_dims)
        self.expert_num=expert_num

        self.expert_model=[MLPLayer(units=[64,8],activation='relu') for _ in range(self.expert_num)]
        self.ctr_gate=MLPLayer(units=[64,self.expert_num],activation='relu')
        self.cvr_gate=MLPLayer(units=[64,self.expert_num],activation='relu')

        self.ctr_output=[MLPLayer(units=[64,8],activation='relu'),MLPLayer(units=[1],activation='sigmoid')]
        self.cvr_output=[MLPLayer(units=[64,8],activation='relu'),MLPLayer(units=[1],activation='sigmoid')]

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        X_cate = []
        for feature in self.categorical_features:
            X_cate.append(inputs[feature])
        X_cate = tf.concat(X_cate, axis=1)

        emb_output = self.embedding_layer(X_cate)
        emb_output=tf.keras.layers.Flatten()(emb_output)


        expert_outputs=[]
        for i in range(self.expert_num):
            expert_output=self.expert_model[i](emb_output)
            expert_outputs.append(expert_output)

        ctr_gate=self.ctr_gate(emb_output)
        cvr_gate=self.cvr_gate(emb_output)

        expert_outputs=tf.stack(expert_outputs,axis=1)  #(batch,expert_num,8)
        ctr_gate=tf.expand_dims(ctr_gate,axis=2)        #(batch,expert_num,1)
        cvr_gate=tf.expand_dims(cvr_gate,axis=2)        #(batch,expert_num,1)
        ctr_input=tf.keras.layers.Flatten()(expert_outputs*ctr_gate)
        cvr_input = tf.keras.layers.Flatten()(expert_outputs * cvr_gate)

        ctr_output=self.ctr_output[0](ctr_input)
        ctr_output=self.ctr_output[1](ctr_output)
        cvr_output = self.cvr_output[0](cvr_input)
        cvr_output = self.cvr_output[1](cvr_output)
        result={'ctr_output':ctr_output,'cvr_output':cvr_output}
        return result

class ESMMLayer(tf.keras.layers.Layer):
    """
        保留MMOE的结构作为基础模型
        mm = ModelManager(layer='esmm_layer',allow_continuous=True)
        input = {'sdk_type': tf.constant([0, 1, 2]), 'remote_host': tf.constant([3, 4, 5]),
                 'device_type': tf.constant([6, 7, 8]), 'dtu': tf.constant([9, 10, 11]),
                 'click_goods_num': tf.constant([12, 13, 14]), 'buy_click_num': tf.constant([15, 16, 17]),
                 'goods_show_num': tf.constant([18, 19, 20]), 'goods_click_num': tf.constant([21, 22, 23]),
                 'brand_name': tf.constant([24, 25, 26]),
                 'click_goods_num_origin':tf.constant([0.2, 7.8, 4.9]),
                 'click_goods_num_square':tf.constant([5.3, 1.2, 8.0]),
                 'click_goods_num_cube': tf.constant([-3.8, -19.6, 4.2])}
        print(mm.model(input))
        """
    def __init__(self,
                 categorical_features=['sdk_type', 'remote_host', 'device_type', 'dtu',
                                       'click_goods_num', 'buy_click_num', 'goods_show_num', 'goods_click_num',
                                       'brand_name'],
                 continuous_features=['click_goods_num_origin', 'click_goods_num_square', 'click_goods_num_cube'], feature_dims=160000,
                 embedding_dims=16, expert_num=3 ):
        super(ESMMLayer, self).__init__()
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims, embedding_dims)
        self.expert_num=expert_num

        self.expert_model=[MLPLayer(units=[64,8],activation='relu') for _ in range(self.expert_num)]
        self.ctr_gate=MLPLayer(units=[64,self.expert_num],activation='relu')
        self.cvr_gate=MLPLayer(units=[64,self.expert_num],activation='relu')

        self.ctr_output=[MLPLayer(units=[64,8],activation='relu'),MLPLayer(units=[1],activation='sigmoid')]
        self.cvr_output=[MLPLayer(units=[64,8],activation='relu'),MLPLayer(units=[1],activation='sigmoid')]

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        X_cate = []
        for feature in self.categorical_features:
            X_cate.append(inputs[feature])
        X_cate = tf.concat(X_cate, axis=1)

        emb_output = self.embedding_layer(X_cate)
        emb_output=tf.keras.layers.Flatten()(emb_output)


        expert_outputs=[]
        for i in range(self.expert_num):
            expert_output=self.expert_model[i](emb_output)
            expert_outputs.append(expert_output)

        ctr_gate=self.ctr_gate(emb_output)
        cvr_gate=self.cvr_gate(emb_output)

        expert_outputs=tf.stack(expert_outputs,axis=1)  #(batch,expert_num,8)
        ctr_gate = tf.nn.softmax(ctr_gate, axis=1)
        cvr_gate = tf.nn.softmax(cvr_gate, axis=1)
        ctr_gate=tf.expand_dims(ctr_gate,axis=2)        #(batch,expert_num,1)
        cvr_gate=tf.expand_dims(cvr_gate,axis=2)        #(batch,expert_num,1)
        ctr_input=tf.keras.layers.Flatten()(expert_outputs*ctr_gate)
        cvr_input = tf.keras.layers.Flatten()(expert_outputs * cvr_gate)

        ctr_output=self.ctr_output[0](ctr_input)
        ctr_output=self.ctr_output[1](ctr_output)
        cvr_output = self.cvr_output[0](cvr_input)
        cvr_output = self.cvr_output[1](cvr_output)
        ctcvr_output=tf.multiply(ctr_output,cvr_output)
        result={'ctr_output':ctr_output,'cvr_output':ctcvr_output}
        return result


class PLELayer(tf.keras.layers.Layer):
    """
        mm = ModelManager(layer='ple_layer',allow_continuous=True)
        input = {'sdk_type': tf.constant([0, 1, 2]), 'remote_host': tf.constant([3, 4, 5]),
                 'device_type': tf.constant([6, 7, 8]), 'dtu': tf.constant([9, 10, 11]),
                 'click_goods_num': tf.constant([12, 13, 14]), 'buy_click_num': tf.constant([15, 16, 17]),
                 'goods_show_num': tf.constant([18, 19, 20]), 'goods_click_num': tf.constant([21, 22, 23]),
                 'brand_name': tf.constant([24, 25, 26]),
                 'click_goods_num_origin':tf.constant([0.2, 7.8, 4.9]),
                 'click_goods_num_square':tf.constant([5.3, 1.2, 8.0]),
                 'click_goods_num_cube': tf.constant([-3.8, -19.6, 4.2])}
        print(mm.model(input))
        """
    def __init__(self,
                 categorical_features=['sdk_type', 'remote_host', 'device_type', 'dtu',
                                       'click_goods_num', 'buy_click_num', 'goods_show_num', 'goods_click_num',
                                       'brand_name'],
                 continuous_features=['click_goods_num_origin', 'click_goods_num_square', 'click_goods_num_cube'], feature_dims=160000,
                 embedding_dims=8, specific_expert_num=3,shared_expert_num=3,level_num=2):
        super(PLELayer, self).__init__()
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims, embedding_dims)
        self.num_tasks=3    # ctr+ctcvr
        self.specific_expert_num=specific_expert_num
        self.shared_expert_num=shared_expert_num
        self.level_num=level_num

        self.specific_expert_networks=[]
        self.shared_expert_networks=[]
        self.specific_gates=[]
        self.shared_gates=[]
        self.tower_outputs=[]

    def build(self, input_shape):
        # 所有需要用到的MLP网络，预先定义并添加到指定的列表中
        self.built = True
        # 1.任务专用的专家网络，每层/每个任务/每个专家有一个MLP
        for level in range(self.level_num):
            level_network=[]
            for i in range(self.num_tasks):
                for j in range(self.specific_expert_num):
                    network=MLPLayer(units=[64,8],activation='relu')
                    level_network.append(network)
            self.specific_expert_networks.append(level_network)
        # 2.共享用的专家网络，每层/每个专家有一个MLP
        for level in range(self.level_num):
            level_network=[]
            for k in range(self.shared_expert_num):
                network = MLPLayer(units=[64, 8], activation='relu')
                level_network.append(network)
            self.shared_expert_networks.append(level_network)
        # 3.任务专用的门网络，每层/每个任务有一个MLP，输出self.specific_expert_num个单元
        for level in range(self.level_num):
            level_gate=[]
            for task in range(self.num_tasks):       # 每一层都输入num_tasks个结果
                gate=[MLPLayer(units=[64,8],activation='relu'),MLPLayer(units=[self.specific_expert_num+self.shared_expert_num],use_bias=False,activation='softmax')]
                level_gate.append(gate)
            self.specific_gates.append(level_gate)
        # 4.共享用的门网络，除最后一层的每层有一个MLP，输出self.shared_expert_num个单元
        for level in range(self.level_num-1):        # 最后一层不输出shared的结果
            gate = [MLPLayer(units=[64, 8], activation='relu'),MLPLayer(units=[self.num_tasks*self.specific_expert_num+self.shared_expert_num], use_bias=False, activation='softmax')]
            self.shared_gates.append(gate)
        # 5.最后的输出MLP,只输出一个单元，有self.num_tasks个MLP
        for task in range(self.num_tasks):
            self.tower_outputs.append(MLPLayer(units=[1], activation='sigmoid'))

    def call_cgc_net(self,inputs,level_num,is_last=False):
        """
        单层cgc网络的调用函数
        """
        outputs=[]
        specific_expert_networks=self.specific_expert_networks[level_num]
        shared_expert_networks=self.shared_expert_networks[level_num]
        specific_gates=self.specific_gates[level_num]

        specific_expert_outputs=[]
        for i in range(self.num_tasks):
            for j in range(self.specific_expert_num):
                index=i*self.specific_expert_num+j
                output=specific_expert_networks[index](inputs[i])
                specific_expert_outputs.append(output)

        shared_expert_outputs=[]
        for k in range(self.shared_expert_num):
            output=shared_expert_networks[k](inputs[-1])
            shared_expert_outputs.append(output)

        for task_index in range(self.num_tasks):
            cur_experts=specific_expert_outputs[task_index * self.specific_expert_num:(task_index + 1) * self.specific_expert_num] + shared_expert_outputs
            expert_concat = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(cur_experts)
            gate_output=specific_gates[task_index][0](inputs[task_index])
            gate_output = specific_gates[task_index][1](gate_output)
            gate_output = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_output)
            gate_mul_expert = tf.keras.layers.Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
                                                     name=str(level_num) + '_gate_mul_expert_specific_' + str(task_index))([expert_concat, gate_output])
            outputs.append(gate_mul_expert)

        if not is_last:
            shared_gates = self.shared_gates[level_num]
            cur_experts = specific_expert_outputs+ shared_expert_outputs
            expert_concat = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(cur_experts)
            gate_output = shared_gates[0](inputs[-1])
            gate_output = shared_gates[1](gate_output)
            gate_output = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_output)
            gate_mul_expert = tf.keras.layers.Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
                                                     name=str(level_num) + '_gate_mul_expert_shared')([expert_concat, gate_output])
            outputs.append(gate_mul_expert)

        return outputs

    def call(self, inputs):
        X_cate = []
        for feature in self.categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_cate.append(feature_tensor)
        X_cate = tf.concat(X_cate, axis=1)

        emb_output = self.embedding_layer(X_cate)
        emb_output=tf.keras.layers.Flatten()(emb_output)
        ple_inputs=[emb_output]*(self.num_tasks+1)
        for level in range(self.level_num-1):
            ple_outputs=self.call_cgc_net(ple_inputs,level,False)
            ple_inputs=ple_outputs
        # 最后一层
        ple_outputs = self.call_cgc_net(ple_inputs, self.level_num-1, True)

        # 顶层输出
        result = {}
        for i in range(self.num_tasks):
            output = self.tower_outputs[i](ple_outputs[i])
            result[f"task_{i}"] = output

        return result


if __name__ == '__main__':
    #input = tf.random.normal((2,3,8))
    layer=PLELayer()
    input = {'sdk_type': tf.constant([0, 1, 2]), 'remote_host': tf.constant([3, 4, 5]),
             'device_type': tf.constant([6, 7, 8]), 'dtu': tf.constant([9, 10, 11]),
             'click_goods_num': tf.constant([12, 13, 14]), 'buy_click_num': tf.constant([15, 16, 17]),
             'goods_show_num': tf.constant([18, 19, 20]), 'goods_click_num': tf.constant([21, 22, 23]),
             'brand_name': tf.constant([24, 25, 26]),
             'click_goods_num_origin': tf.constant([0.2, 7.8, 4.9]),
             'click_goods_num_square': tf.constant([5.3, 1.2, 8.0]),
             'click_goods_num_cube': tf.constant([-3.8, -19.6, 4.2])}
    print(layer(input))

