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
        mlp.add(tf.keras.layers.Dense(softmax_units, activation='softmax',use_bias=False))
    elif sigmoid_units:
        mlp.add(tf.keras.layers.Dense(1, activation='sigmoid',use_bias=False))
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


class MMOELayer(tf.keras.layers.Layer):
    # ctr和cvr任务，混合训练，还是交替训练，还是先后训练
    # 混合训练，计算损失时，对于ctr_label=0的部分打mask，不进行梯度回传
    # 交替训练，两份数据的epoch交替迭代
    # 先后训练，先训练ctr数据，再训练cvr数据
    """
    inputs = {'sdk_type': tf.constant([0, 1, 2]), 'remote_host': tf.constant([3, 4, 5]),
                 'device_type': tf.constant([6, 7, 8]), 'dtu': tf.constant([9, 10, 11]),
                 'click_goods_num': tf.constant([12, 13, 14]), 'buy_click_num': tf.constant([15, 16, 17]),
                 'goods_show_num': tf.constant([18, 19, 20]), 'goods_click_num': tf.constant([21, 22, 23]),
                 'brand_name': tf.constant([24, 25, 26]),
                 'click_goods_num_origin':tf.constant([0.2, 7.8, 4.9]),
                 'click_goods_num_square':tf.constant([5.3, 1.2, 8.0]),
                 'click_goods_num_cube': tf.constant([-3.8, -19.6, 4.2])}
    layer = MMOELayer()
    set_custom_initialization(layer)
    print(layer(inputs))
    print('****************')
    input_dict = {}
    max_length = 10
    neg_samples=6
    for feature in layer.categorical_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.continuous_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
    output = layer(input_dict)
    model = tf.keras.Model(input_dict, output)
    model.summary()
    print(model(inputs))
    """
    def __init__(self,
                 categorical_features=['sdk_type', 'remote_host', 'device_type', 'dtu',
                                       'click_goods_num', 'buy_click_num', 'goods_show_num', 'goods_click_num',
                                       'brand_name'],
                 continuous_features=['click_goods_num_origin', 'click_goods_num_square', 'click_goods_num_cube'], feature_dims=160000,
                 embedding_dims=8, expert_num=3,expert_model_units=[64,8],gate_units=[64],output_units=[64,8],
                 expert_activation='ReLU',gate_activation='ReLU',output_activation='ReLU'):
        super(MMOELayer, self).__init__()
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims, embedding_dims)
        self.expert_num=expert_num

        # expert/gate/output_model都可以替换成其它结构，只要保证输入/输出单元数一致即可
        # expert_model:MLP
        self.expert_model=[make_mlp_layer(units=expert_model_units,activation=expert_activation) for _ in range(self.expert_num)]

        # gate_model:MLP
        self.ctr_gate=make_mlp_layer(units=gate_units+[expert_num],activation=gate_activation)
        self.cvr_gate=make_mlp_layer(units=gate_units+[expert_num],activation=gate_activation)

        # output_model:MLP
        self.ctr_output=make_mlp_layer(units=output_units,activation=output_activation,sigmoid_units=True)
        self.cvr_output=make_mlp_layer(units=output_units,activation=output_activation,sigmoid_units=True)

    def generate_expert_output(self,inputs):
        X_cate = []
        for feature in self.categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_cate.append(feature_tensor)
        X_cate = tf.concat(X_cate, axis=1)

        emb_output = self.embedding_layer(X_cate)
        emb_output = tf.keras.layers.Flatten()(emb_output)

        X_cont = []
        for feature in self.continuous_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_cont.append(feature_tensor)
        X_cont = tf.concat(X_cont, axis=1)

        X_combined = tf.concat([emb_output, X_cont], axis=1)

        expert_outputs = []
        for i in range(self.expert_num):
            expert_output = self.expert_model[i](X_combined)
            expert_outputs.append(expert_output)
            tf.stack(expert_outputs, axis=1)
        return X_combined,expert_outputs

    def generate_task_output(self,X_combined,expert_outputs,task):
        if task=='ctr':
            gate=self.ctr_gate
            output=self.ctr_output
        else:
            gate=self.cvr_gate
            output=self.cvr_output
        gate = gate(X_combined)
        gate = tf.expand_dims(gate, axis=2)
        input = tf.multiply(expert_outputs, gate)
        input = tf.keras.layers.Flatten()(input)
        output = output(input)
        return output

    def call(self, inputs):
        X_combined,expert_outputs=self.generate_expert_output(inputs)
        ctr_output=self.generate_task_output(X_combined,expert_outputs,task='ctr')
        cvr_output = self.generate_task_output(X_combined, expert_outputs, task='cvr')
        result={'ctr_output':ctr_output,'cvr_output':cvr_output}
        return result


class ESMMLayer(tf.keras.layers.Layer):
    # label:ctr+ctcvr
    # 全曝光数据一起训练，同时更新两个目标
    """
    inputs = {'sdk_type': tf.constant([0, 1, 2]), 'remote_host': tf.constant([3, 4, 5]),
                 'device_type': tf.constant([6, 7, 8]), 'dtu': tf.constant([9, 10, 11]),
                 'click_goods_num': tf.constant([12, 13, 14]), 'buy_click_num': tf.constant([15, 16, 17]),
                 'goods_show_num': tf.constant([18, 19, 20]), 'goods_click_num': tf.constant([21, 22, 23]),
                 'brand_name': tf.constant([24, 25, 26]),
                 'click_goods_num_origin':tf.constant([0.2, 7.8, 4.9]),
                 'click_goods_num_square':tf.constant([5.3, 1.2, 8.0]),
                 'click_goods_num_cube': tf.constant([-3.8, -19.6, 4.2])}
    layer = ESMMLayer()
    set_custom_initialization(layer)
    print(layer(inputs))
    print('****************')
    input_dict = {}
    max_length = 10
    neg_samples=6
    for feature in layer.categorical_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.continuous_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
    output = layer(input_dict)
    model = tf.keras.Model(input_dict, output)
    model.summary()
    print(model(inputs))
    """
    def __init__(self,
                 categorical_features=['sdk_type', 'remote_host', 'device_type', 'dtu',
                                       'click_goods_num', 'buy_click_num', 'goods_show_num', 'goods_click_num',
                                       'brand_name'],
                 continuous_features=['click_goods_num_origin', 'click_goods_num_square', 'click_goods_num_cube'],
                 feature_dims=160000,
                 embedding_dims=8, expert_num=3, expert_model_units=[64, 8], gate_units=[64], output_units=[64, 8],
                 expert_activation='ReLU', gate_activation='ReLU', output_activation='ReLU'):
        super(ESMMLayer, self).__init__()
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims, embedding_dims)
        self.expert_num = expert_num

        # expert/gate/output_model都可以替换成其它结构，只要保证输入/输出单元数一致即可
        # expert_model:MLP
        self.expert_model = [make_mlp_layer(units=expert_model_units, activation=expert_activation) for _ in
                             range(self.expert_num)]

        # gate_model:MLP
        self.ctr_gate = make_mlp_layer(units=gate_units + [expert_num], activation=gate_activation)
        self.cvr_gate = make_mlp_layer(units=gate_units + [expert_num], activation=gate_activation)

        # output_model:MLP
        self.ctr_output = make_mlp_layer(units=output_units, activation=output_activation, sigmoid_units=True)
        self.cvr_output = make_mlp_layer(units=output_units, activation=output_activation, sigmoid_units=True)

    def generate_expert_output(self, inputs):
        X_cate = []
        for feature in self.categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_cate.append(feature_tensor)
        X_cate = tf.concat(X_cate, axis=1)

        emb_output = self.embedding_layer(X_cate)
        emb_output = tf.keras.layers.Flatten()(emb_output)

        X_cont = []
        for feature in self.continuous_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_cont.append(feature_tensor)
        X_cont = tf.concat(X_cont, axis=1)

        X_combined = tf.concat([emb_output, X_cont], axis=1)

        expert_outputs = []
        for i in range(self.expert_num):
            expert_output = self.expert_model[i](X_combined)
            expert_outputs.append(expert_output)
            tf.stack(expert_outputs, axis=1)
        return X_combined, expert_outputs

    def generate_task_output(self, X_combined, expert_outputs, task):
        if task == 'ctr':
            gate = self.ctr_gate
            output = self.ctr_output
        else:
            gate = self.cvr_gate
            output = self.cvr_output
        gate = gate(X_combined)
        gate=tf.nn.softmax(gate,axis=-1)
        gate = tf.expand_dims(gate, axis=2)
        input = tf.multiply(expert_outputs, gate)
        input = tf.keras.layers.Flatten()(input)
        output = output(input)
        return output

    def call(self, inputs):
        X_combined, expert_outputs = self.generate_expert_output(inputs)
        ctr_output = self.generate_task_output(X_combined, expert_outputs, task='ctr')
        cvr_output = self.generate_task_output(X_combined, expert_outputs, task='cvr')
        ctcvr_output=ctr_output*cvr_output
        result = {'ctr_output': ctr_output, 'ctcvr_output': ctcvr_output}
        return result


class PLELayer(tf.keras.layers.Layer):
    """
    inputs = {'sdk_type': tf.constant([0, 1, 2]), 'remote_host': tf.constant([3, 4, 5]),
                 'device_type': tf.constant([6, 7, 8]), 'dtu': tf.constant([9, 10, 11]),
                 'click_goods_num': tf.constant([12, 13, 14]), 'buy_click_num': tf.constant([15, 16, 17]),
                 'goods_show_num': tf.constant([18, 19, 20]), 'goods_click_num': tf.constant([21, 22, 23]),
                 'brand_name': tf.constant([24, 25, 26]),
                 'click_goods_num_origin':tf.constant([0.2, 7.8, 4.9]),
                 'click_goods_num_square':tf.constant([5.3, 1.2, 8.0]),
                 'click_goods_num_cube': tf.constant([-3.8, -19.6, 4.2])}
    layer = PLELayer()
    set_custom_initialization(layer)
    print(layer(inputs))
    print('****************')
    input_dict = {}
    max_length = 10
    neg_samples=6
    for feature in layer.categorical_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.continuous_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
    output = layer(input_dict)
    model = tf.keras.Model(input_dict, output)
    model.summary()
    print(model(inputs))
    """
    def __init__(self,
                 categorical_features=['sdk_type', 'remote_host', 'device_type', 'dtu',
                                       'click_goods_num', 'buy_click_num', 'goods_show_num', 'goods_click_num',
                                       'brand_name'],
                 continuous_features=['click_goods_num_origin', 'click_goods_num_square', 'click_goods_num_cube'], feature_dims=160000,
                 embedding_dims=8, specific_expert_num=4,shared_expert_num=4,level_num=3,num_tasks=3,activation='ReLU',
                 expert_units=[64,8],gate_units=[64,8],return_cgc_output=False):
        super(PLELayer, self).__init__()
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims, embedding_dims)
        self.num_tasks=num_tasks    # ctr+ctcvr
        self.specific_expert_num=specific_expert_num
        self.shared_expert_num=shared_expert_num
        self.level_num=level_num
        self.activation=activation
        self.expert_units=expert_units
        self.gate_units=gate_units

        self.specific_expert_networks=[]
        self.shared_expert_networks=[]
        self.specific_gates=[]
        self.shared_gates=[]
        self.tower_outputs=[]

        self.return_cgc_output=return_cgc_output

    def build(self, input_shape):
        # 所有需要用到的MLP网络，预先定义并添加到指定的列表中
        self.built = True
        # 1.任务专用的专家网络，每层/每个任务/每个专家有一个MLP
        for level in range(self.level_num):
            level_network=[]
            for i in range(self.num_tasks):
                for j in range(self.specific_expert_num):
                    network=make_mlp_layer(units=self.expert_units,activation=self.activation)
                    level_network.append(network)
            self.specific_expert_networks.append(level_network)
        # 2.共享用的专家网络，每层/每个专家有一个MLP
        for level in range(self.level_num):
            level_network=[]
            for k in range(self.shared_expert_num):
                network = make_mlp_layer(units=self.expert_units, activation=self.activation)
                level_network.append(network)
            self.shared_expert_networks.append(level_network)
        # 3.任务专用的门网络，每层/每个任务有一个MLP，输出self.specific_expert_num个单元
        for level in range(self.level_num):
            level_gate=[]
            for task in range(self.num_tasks):       # 每一层都输入num_tasks个结果
                gate=make_mlp_layer(units=self.gate_units,activation=self.activation,softmax_units=self.specific_expert_num+self.shared_expert_num)
                level_gate.append(gate)
            self.specific_gates.append(level_gate)
        # 4.共享用的门网络，除最后一层的每层有一个MLP，输出self.shared_expert_num个单元
        for level in range(self.level_num-1):        # 最后一层不输出shared的结果
            gate = make_mlp_layer(units=self.gate_units, activation=self.activation,
                                  softmax_units=self.num_tasks*self.specific_expert_num+self.shared_expert_num)
            self.shared_gates.append(gate)
        # 5.最后的输出MLP,只输出一个单元，有self.num_tasks个MLP
        for task in range(self.num_tasks):
            self.tower_outputs.append(make_mlp_layer([],sigmoid_units=True))

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
            expert_concat = tf.stack(cur_experts, axis=1)  # (batch_size,expert_num,expert_dim)
            gate_output=specific_gates[task_index](inputs[task_index])
            gate_output = tf.expand_dims(gate_output, axis=-1)
            gate_mul_expert = tf.reduce_sum(tf.multiply(expert_concat,gate_output),axis=1) # (batch_size,expert_dim)
            outputs.append(gate_mul_expert)

        if not is_last:
            shared_gates = self.shared_gates[level_num]
            cur_experts = specific_expert_outputs+ shared_expert_outputs
            expert_concat = tf.stack(cur_experts, axis=1)
            gate_output = shared_gates(inputs[-1])
            gate_output = tf.expand_dims(gate_output, axis=-1)
            gate_mul_expert = tf.reduce_sum(tf.multiply(expert_concat, gate_output), axis=1)  # (batch_size,expert_dim)
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
        result={}
        for i in range(self.num_tasks):
            output=self.tower_outputs[i](ple_outputs[i])
            result[f"task_{i}"]=output
        if self.return_cgc_output:
            result['cgc_output']=ple_outputs
        return result


class ESM2Layer(tf.keras.layers.Layer):
    """
    task0:click->task1:cart     -->task5:order-->task8:pay
                 task2:collect  -->task6:order
                 task3:none     -->task7:order

    inputs = {'sdk_type': tf.constant([0, 1, 2,3, 4, 5]), 'remote_host': tf.constant([3, 4, 5,6, 7, 8]),
             'device_type': tf.constant([9, 10, 11,12, 13, 14]), 'dtu': tf.constant([15, 16, 17,18, 19, 20]),
             'click_goods_num': tf.constant([21, 22, 23,24, 25, 26]), 'buy_click_num': tf.constant([27,28,29,30,31,32]),
             'goods_show_num': tf.constant([33,34,35,36,37,38]), 'goods_click_num': tf.constant([39,40,41,42,43,44]),
             'brand_name': tf.constant([45,46,47,48,49,50]),
             'click_goods_num_origin': tf.constant([0.2, 7.8, 4.9,1.8,1.9,2.3]),
             'click_goods_num_square': tf.constant([5.3, 1.2, 8.0,7.8,-0.2,4.2]),
             'click_goods_num_cube': tf.constant([-3.8, -19.6, 4.2,9.7,0.2,-7.3])}
    inputs.update({'label_0':tf.constant([1,1,1,1,1,0]),
                   'label_1':tf.constant([1,0,0,0,0,0]),
                   'label_2':tf.constant([0,1,0,0,0,0]),
                   'label_3':tf.constant([0,0,1,0,0,0]),
                   'label_4':tf.constant([1,0,1,1,0,0]),
                   'label_5':tf.constant([1,0,1,0,0,0])})
    layer = ESM2Layer()
    set_custom_initialization(layer)
    print(layer(inputs))
    print('****************')
    input_dict = {}
    max_length = 10
    neg_samples=6
    for feature in layer.categorical_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.continuous_features+[f'label_{i}' for i in range(6)]:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
    output = layer(input_dict)
    model = tf.keras.Model(input_dict, output)
    model.summary()
    print(model(inputs))
    """
    def __init__(self,
                 categorical_features=['sdk_type', 'remote_host', 'device_type', 'dtu',
                                       'click_goods_num', 'buy_click_num', 'goods_show_num', 'goods_click_num',
                                       'brand_name'],
                 continuous_features=['click_goods_num_origin', 'click_goods_num_square', 'click_goods_num_cube'],
                 feature_dims=160000,
                 embedding_dims=8, expert_num=3, expert_model_units=[64, 8], gate_units=[64], output_units=[64, 8],
                 expert_activation='ReLU', gate_activation='ReLU', output_activation='ReLU',num_tasks=8,
                 weight0=1,weight1=1,weight2=1,weight3=1,weight4=1,weight5=1):
        super(ESM2Layer, self).__init__()
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims, embedding_dims)
        self.expert_num = expert_num
        self.num_tasks=num_tasks
        self.weight_list=[weight0,weight1,weight2,weight3,weight4,weight5]

        # expert/gate/output_model都可以替换成其它结构，只要保证输入/输出单元数一致即可
        # expert_model:MLP
        self.expert_model = [make_mlp_layer(units=expert_model_units, activation=expert_activation) for _ in
                             range(expert_num)]

        # gate_model:MLP
        self.expert_gates = [make_mlp_layer(units=gate_units + [expert_num], activation=gate_activation) for _ in range(num_tasks)]

        # output_model:MLP
        self.output_layers = [make_mlp_layer(units=output_units, activation=output_activation, sigmoid_units=True) for _ in range(num_tasks)]

    def generate_expert_output(self, inputs):
        X_cate = []
        for feature in self.categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_cate.append(feature_tensor)
        X_cate = tf.concat(X_cate, axis=1)

        emb_output = self.embedding_layer(X_cate)
        emb_output = tf.keras.layers.Flatten()(emb_output)

        X_cont = []
        for feature in self.continuous_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_cont.append(feature_tensor)
        X_cont = tf.concat(X_cont, axis=1)

        X_combined = tf.concat([emb_output, X_cont], axis=1)

        expert_outputs = []
        for i in range(self.expert_num):
            expert_output = self.expert_model[i](X_combined)
            expert_outputs.append(expert_output)
        expert_outputs=tf.stack(expert_outputs, axis=1)
        return X_combined, expert_outputs

    def generate_task_output(self, X_combined, expert_outputs, i):
        gate = self.expert_gates[i]
        output = self.output_layers[i]
        gate = gate(X_combined)
        gate=tf.nn.softmax(gate,axis=-1)
        gate = tf.expand_dims(gate, axis=2)
        input = tf.multiply(expert_outputs, gate)
        input = tf.keras.layers.Flatten()(input)
        output = output(input)
        return output

    def call(self, inputs):
        X_combined, expert_outputs = self.generate_expert_output(inputs)
        result={}
        for i in range(self.num_tasks):
            output=self.generate_task_output(X_combined, expert_outputs, i)
            result[f"task_{i}"]=output
        # task0:expose-->click
        # task1:click-->cart
        # task2:click-->collect
        # task3:click-->none
        # task4:cart-->order
        # task5:collect-->order
        # task6:none-->order
        # task7:order-->pay
        click_label=tf.reshape(tf.cast(inputs['label_0'],dtype=tf.float32),[-1,1])
        click_probs=result['task_0']
        click_loss=tf.keras.backend.binary_crossentropy(click_label,click_probs,from_logits=False)
        click_loss=tf.reduce_mean(click_loss)
        #print(click_probs.shape)
        # click_loss = - (tf.cast(click_label,dtype=tf.float32) * tf.math.log(click_probs) + (1 - tf.cast(click_label,dtype=tf.float32)) * tf.math.log(1 - click_probs))
        # click_loss = tf.reduce_mean(click_loss)

        cart_label=tf.reshape(tf.cast(inputs['label_1'],dtype=tf.float32),[-1,1])
        cart_probs= click_probs*result['task_1']
        cart_loss = tf.keras.backend.binary_crossentropy(cart_label,cart_probs, from_logits=False)
        cart_loss = tf.reduce_mean(cart_loss)

        # label中优先cart，cart覆盖collect
        collect_label = tf.reshape(tf.cast(inputs['label_2'],dtype=tf.float32),[-1,1])
        collect_probs = click_probs * (1-result['task_1'])*result['task_2']
        collect_loss = tf.keras.backend.binary_crossentropy(tf.cast(collect_label, dtype=tf.float32),
                                                         tf.reshape(collect_probs, [-1]), from_logits=False)
        collect_loss = tf.reduce_mean(collect_loss)

        # cart>collect>none
        none_label=tf.reshape(tf.cast(inputs['label_3'],dtype=tf.float32),[-1,1])
        none_probs = click_probs * (1-result['task_1'])*(1-result['task_2'])* result['task_3']
        none_loss = tf.keras.backend.binary_crossentropy(tf.cast(none_label, dtype=tf.float32),
                                                            tf.reshape(none_probs, [-1]), from_logits=False)
        none_loss = tf.reduce_mean(none_loss)

        order_label = tf.reshape(tf.cast(inputs['label_4'],dtype=tf.float32),[-1,1])
        order_probs = cart_probs*result['task_4']+collect_probs*result['task_5']+none_probs*result['task_6']
        order_loss = tf.keras.backend.binary_crossentropy(tf.cast(order_label, dtype=tf.float32),
                                                         tf.reshape(order_probs, [-1]), from_logits=False)
        order_loss = tf.reduce_mean(order_loss)

        pay_label = tf.reshape(tf.cast(inputs['label_5'],dtype=tf.float32),[-1,1])
        pay_probs = order_probs*result['task_5']
        pay_loss = tf.keras.backend.binary_crossentropy(tf.cast(pay_label, dtype=tf.float32),
                                                          tf.reshape(pay_probs, [-1]), from_logits=False)
        pay_loss = tf.reduce_mean(pay_loss)

        total_loss=click_loss*self.weight_list[0]+\
                   cart_loss*self.weight_list[1]+\
                   collect_loss*self.weight_list[2]+\
                   none_loss*self.weight_list[3]+\
                   order_loss*self.weight_list[4]+\
                   pay_loss*self.weight_list[5]
        return total_loss


class ESCM2Layer(tf.keras.layers.Layer):
    """
    inputs = {'sdk_type': tf.constant([0, 1, 2,3, 4, 5]), 'remote_host': tf.constant([3, 4, 5,6, 7, 8]),
             'device_type': tf.constant([9, 10, 11,12, 13, 14]), 'dtu': tf.constant([15, 16, 17,18, 19, 20]),
             'click_goods_num': tf.constant([21, 22, 23,24, 25, 26]), 'buy_click_num': tf.constant([27,28,29,30,31,32]),
             'goods_show_num': tf.constant([33,34,35,36,37,38]), 'goods_click_num': tf.constant([39,40,41,42,43,44]),
             'brand_name': tf.constant([45,46,47,48,49,50]),
             'click_goods_num_origin': tf.constant([0.2, 7.8, 4.9,1.8,1.9,2.3]),
             'click_goods_num_square': tf.constant([5.3, 1.2, 8.0,7.8,-0.2,4.2]),
             'click_goods_num_cube': tf.constant([-3.8, -19.6, 4.2,9.7,0.2,-7.3])}
    inputs.update({'label_ctr': tf.constant([1, 1, 1, 1, 0, 0]),
                   'label_ctcvr': tf.constant([1, 0, 1, 0, 0, 0]),
                   })
    layer = ESCM2Layer(counterfactual_mode='IPS')
    set_custom_initialization(layer)
    print(layer(inputs))
    print('****************')
    input_dict = {}
    for feature in layer.base_model.categorical_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.base_model.continuous_features+[layer.ctr_label_name,layer.ctcvr_label_name]:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
    output = layer(input_dict)
    model = tf.keras.Model(input_dict, output)
    model.summary()
    print(model(inputs))
    """
    def __init__(self,
                 base_model='ple',model_params={},ctr_label_name='label_ctr',ctcvr_label_name='label_ctcvr',
                 counterfactual_w=0.5,global_w=0.5,counterfactual_mode='DR',dr_mlp_units=[16]):
        super(ESCM2Layer, self).__init__()
        if base_model=='ple':
            self.base_model=PLELayer(num_tasks=2,return_cgc_output=True,**model_params)
            self.ctr_predict_name='task_0'
            self.cvr_predict_name='task_1'
        elif base_model=='mmoe':
            self.base_model=MMOELayer()
            self.ctr_predict_name='ctr_output'
            self.cvr_predict_name='cvr_output'
        self.ctr_label_name=ctr_label_name
        self.ctcvr_label_name=ctcvr_label_name
        self.counterfactual_w=counterfactual_w
        self.global_w=global_w
        self.counterfactual_mode=counterfactual_mode
        if counterfactual_mode=='DR':
            self.dr_mlp=make_mlp_layer(dr_mlp_units,activation='ReLU',sigmoid_units=True)

    def call(self, inputs):
        # 使用的交叉熵计算api要求乘法两边数据类型一致
        ctr_label=tf.cast(inputs[self.ctr_label_name],dtype=tf.float32)
        ctcvr_label=tf.cast(inputs[self.ctcvr_label_name],dtype=tf.float32)
        # 点击域的ctcvr_label:在第一维展开，以取得等于1的索引，然后再还原
        flattened_ctr_label=tf.reshape(ctr_label,[-1])
        indices = tf.where(flattened_ctr_label == 1)
        clicked_ctcvr_label= tf.squeeze(tf.gather(tf.reshape(ctcvr_label,[-1]), indices),axis=-1)
        # label统一转换成(batch_size，1)
        ctr_label=tf.reshape(ctr_label,[-1,1])
        ctcvr_label=tf.reshape(ctcvr_label,[-1,1])
        clicked_ctcvr_label=tf.reshape(clicked_ctcvr_label,[-1,1])

        outputs=self.base_model(inputs)
        ctr_predict=outputs[self.ctr_predict_name]
        cvr_predict=outputs[self.cvr_predict_name]
        ctcvr_predict=tf.squeeze(tf.gather(cvr_predict*ctr_predict, indices),axis=-1)
        loss_ctr = tf.keras.backend.binary_crossentropy(
            target=ctr_label, output=ctr_predict,from_logits=False)
        loss_cvr = tf.keras.backend.binary_crossentropy(
            target=ctcvr_label, output=cvr_predict,from_logits=False)
        loss_ctcvr = tf.keras.backend.binary_crossentropy(
            target=clicked_ctcvr_label,output=ctcvr_predict,from_logits=False)

        if self.counterfactual_mode=='DR':
            cgc_output = tf.concat(outputs['cgc_output'],axis=1)
            imp_out=self.dr_mlp(cgc_output)
            loss_cvr=self.counterfact_dr(loss_cvr,ctr_predict,ctr_label,imp_out)
        elif self.counterfactual_mode=='IPS':
            loss_cvr = self.counterfact_ips(loss_cvr,ctr_predict,ctr_label)
        # print(tf.reduce_mean(loss_ctr))
        # print(loss_cvr)
        # print(tf.reduce_mean(loss_ctcvr))
        loss=tf.reduce_mean(loss_ctr)+self.counterfactual_w*loss_cvr+self.global_w*tf.reduce_mean(loss_ctcvr)
        return loss

    def counterfact_dr(self,loss_cvr,ctr_predict,ctr_label,imp_out):
        # dr error part
        e=loss_cvr-imp_out
        ctr_predict=tf.clip_by_value(ctr_predict,1e-6,1)
        loss_error_second=ctr_label*e/ctr_predict
        loss_error=imp_out+loss_error_second

        # dr imp part
        loss_imp=ctr_label*tf.square(e)/ctr_predict
        loss=tf.reduce_mean(loss_error+loss_imp)
        return loss

    def counterfact_ips(self,loss_cvr,ctr_predict,ctr_label):
        ctr_num=tf.cast(tf.shape(ctr_label)[0],dtype=tf.float32)
        IPS=tf.clip_by_value(ctr_num*ctr_predict,1e-6,1)
        IPS=tf.stop_gradient(IPS)
        loss=tf.reduce_mean(ctr_label*loss_cvr/IPS)
        return loss


class FDN4PLELayer(PLELayer):
    # 要求task_expert和shared_expert成对进行正交约束;专家网络通过简单MLP能一定程度上直接预测目标;主流程逻辑不变
    """
    inputs = {'sdk_type': tf.constant([0, 1, 2,3, 4, 5]), 'remote_host': tf.constant([3, 4, 5,6, 7, 8]),
             'device_type': tf.constant([9, 10, 11,12, 13, 14]), 'dtu': tf.constant([15, 16, 17,18, 19, 20]),
             'click_goods_num': tf.constant([21, 22, 23,24, 25, 26]), 'buy_click_num': tf.constant([27,28,29,30,31,32]),
             'goods_show_num': tf.constant([33,34,35,36,37,38]), 'goods_click_num': tf.constant([39,40,41,42,43,44]),
             'brand_name': tf.constant([45,46,47,48,49,50]),
             'click_goods_num_origin': tf.constant([0.2, 7.8, 4.9,1.8,1.9,2.3]),
             'click_goods_num_square': tf.constant([5.3, 1.2, 8.0,7.8,-0.2,4.2]),
             'click_goods_num_cube': tf.constant([-3.8, -19.6, 4.2,9.7,0.2,-7.3])}
    inputs.update({'label_0': tf.constant([1, 1, 1, 1, 0, 0]),
                   'label_1': tf.constant([1, 0, 1, 0, 0, 0]),
                   })
    layer = FCN4PLELayer()
    set_custom_initialization(layer)
    print(layer(inputs))
    print('****************')
    input_dict = {}
    for feature in layer.categorical_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.continuous_features+['label_0','label_1']:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
    output = layer(input_dict)
    model = tf.keras.Model(input_dict, output)
    model.summary()
    print(model(inputs))
    layer.set_inference(True)
    print(model(inputs))
    """
    def __init__(self,
                 categorical_features=['sdk_type', 'remote_host', 'device_type', 'dtu',
                                       'click_goods_num', 'buy_click_num', 'goods_show_num', 'goods_click_num',
                                       'brand_name'],
                 continuous_features=['click_goods_num_origin', 'click_goods_num_square', 'click_goods_num_cube'],
                 feature_dims=160000,
                 embedding_dims=8, specific_expert_num=4, level_num=3, num_tasks=2,
                 activation='ReLU',
                 expert_units=[64, 8], gate_units=[64, 8], return_cgc_output=False,inference=False):
        super().__init__(categorical_features,continuous_features,feature_dims,
                 embedding_dims, specific_expert_num, specific_expert_num*num_tasks, level_num, num_tasks,
                 activation,expert_units, gate_units, return_cgc_output)
        # 规定shared_expert_num=specific_expert_num*num_tasks
        self.inference=inference

    def build(self,input_shape):
        super().build(input_shape)
        self.aux_mlp_list=[[make_mlp_layer([8],activation='ReLU',sigmoid_units=True) for _ in range(self.specific_expert_num)] for _ in range(self.level_num)]

    def set_inference(self,value):
        self.inference=value

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
            expert_concat = tf.stack(cur_experts, axis=1)  # (batch_size,expert_num,expert_dim)
            gate_output=specific_gates[task_index](inputs[task_index])
            gate_output = tf.expand_dims(gate_output, axis=-1)
            gate_mul_expert = tf.reduce_sum(tf.multiply(expert_concat,gate_output),axis=1) # (batch_size,expert_dim)
            outputs.append(gate_mul_expert)

        if not is_last:
            shared_gates = self.shared_gates[level_num]
            cur_experts = specific_expert_outputs+ shared_expert_outputs
            expert_concat = tf.stack(cur_experts, axis=1)
            gate_output = shared_gates(inputs[-1])
            gate_output = tf.expand_dims(gate_output, axis=-1)
            gate_mul_expert = tf.reduce_sum(tf.multiply(expert_concat, gate_output), axis=1)  # (batch_size,expert_dim)
            outputs.append(gate_mul_expert)

        return outputs,specific_expert_outputs,shared_expert_outputs

    def call(self, inputs):
        X_cate = []
        for feature in self.categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_cate.append(feature_tensor)
        X_cate = tf.concat(X_cate, axis=1)

        label_task0=tf.reshape(tf.cast(inputs['label_0'],dtype=tf.float32),[-1])
        label_task1=tf.reshape(tf.cast(inputs['label_1'],dtype=tf.float32),[-1])

        tasks_labels=[label_task0]*self.specific_expert_num+[label_task1]*self.specific_expert_num
        total_orth_loss=0
        total_aux_loss=0

        emb_output = self.embedding_layer(X_cate)
        emb_output=tf.keras.layers.Flatten()(emb_output)
        ple_inputs=[emb_output]*(self.num_tasks+1)
        for level in range(self.level_num-1):
            ple_outputs,specific_expert_outputs,shared_expert_outputs=self.call_cgc_net(ple_inputs,level,False)
            orth_loss=self.build_orth_loss(specific_expert_outputs,shared_expert_outputs) if not self.inference else 0
            aux_loss=self.build_level_aux_loss(specific_expert_outputs,level,tasks_labels) if not self.inference else 0
            total_orth_loss+=orth_loss
            total_aux_loss+=aux_loss
            ple_inputs=ple_outputs
        # 最后一层
        ple_outputs,specific_expert_outputs,shared_expert_outputs = self.call_cgc_net(ple_inputs, self.level_num-1, True)
        orth_loss = self.build_orth_loss(specific_expert_outputs, shared_expert_outputs) if not self.inference else 0
        aux_loss = self.build_level_aux_loss(specific_expert_outputs, self.level_num-1, tasks_labels) if not self.inference else 0
        total_orth_loss += orth_loss
        total_aux_loss += aux_loss

        # 顶层输出
        result={}
        total_main_loss=0
        labels=[label_task0,label_task1]
        for i in range(self.num_tasks):
            output=tf.reshape(self.tower_outputs[i](ple_outputs[i]),[-1])
            main_loss=tf.reduce_mean(tf.keras.backend.binary_crossentropy(target=labels[i], output=output,from_logits=False))
            total_main_loss+=main_loss
            result[f"task_{i}"]=output
        if self.return_cgc_output:
            result['cgc_output']=ple_outputs
        return total_main_loss,total_aux_loss,total_orth_loss

    def build_orth_loss(self,vector_a,vector_b):
        # 要求a中每列和b中每列都尽量正交，此时其内积都是0
        vector_a=tf.stack(vector_a,axis=1)  # (batch_size,m,embedding_dim)
        vector_b=tf.stack(vector_b,axis=1)  # (batch_size,m,embedding_dim)
        inner_products=tf.einsum('bme,bme->bm',vector_a,vector_b)
        squared_products=tf.square(inner_products)
        orth_loss=tf.reduce_mean(tf.reduce_sum(squared_products,axis=1))
        return orth_loss

    def build_level_aux_loss(self,specific_expert_outputs,level_num,task_labels):
        # 要求task_specific_experts一定程度上要具备直接预测task目标的能力
        total_loss=0
        for expert,net,task_label in zip(specific_expert_outputs,self.aux_mlp_list[level_num],task_labels):
            output=tf.reshape(net(expert),[-1])
            loss=tf.reduce_mean(tf.keras.backend.binary_crossentropy(target=task_label, output=output,from_logits=False))
            total_loss+=loss
        return total_loss


if __name__ == '__main__':
    inputs = {'sdk_type': tf.constant([0, 1, 2, 3, 4, 5]), 'remote_host': tf.constant([3, 4, 5, 6, 7, 8]),
              'device_type': tf.constant([9, 10, 11, 12, 13, 14]), 'dtu': tf.constant([15, 16, 17, 18, 19, 20]),
              'click_goods_num': tf.constant([21, 22, 23, 24, 25, 26]),
              'buy_click_num': tf.constant([27, 28, 29, 30, 31, 32]),
              'goods_show_num': tf.constant([33, 34, 35, 36, 37, 38]),
              'goods_click_num': tf.constant([39, 40, 41, 42, 43, 44]),
              'brand_name': tf.constant([45, 46, 47, 48, 49, 50]),
              'click_goods_num_origin': tf.constant([0.2, 7.8, 4.9, 1.8, 1.9, 2.3]),
              'click_goods_num_square': tf.constant([5.3, 1.2, 8.0, 7.8, -0.2, 4.2]),
              'click_goods_num_cube': tf.constant([-3.8, -19.6, 4.2, 9.7, 0.2, -7.3])}
    inputs.update({'label_ctr': tf.constant([1, 1, 1, 1, 0, 0]),
                   'label_ctcvr': tf.constant([1, 0, 1, 0, 0, 0]),
                   })
    layer = ESCM2Layer(counterfactual_mode='IPS')
    set_custom_initialization(layer)
    print(layer(inputs))
    print('****************')
    input_dict = {}
    for feature in layer.base_model.categorical_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.base_model.continuous_features + [layer.ctr_label_name, layer.ctcvr_label_name]:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
    output = layer(input_dict)
    model = tf.keras.Model(input_dict, output)
    model.summary()
    print(model(inputs))

