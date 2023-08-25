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


class GateNULayer(tf.keras.layers.Layer):
    """
    input=tf.random.normal((3,4,5))
    layer=GateNULayer()
    print(layer(input))
    """
    def __init__(self,hidden_unit=16,output_unit=16):
        super().__init__()
        self.gate=tf.keras.models.Sequential()
        self.gate.add(tf.keras.layers.Dense(hidden_unit,activation='relu'))
        self.gate.add(tf.keras.layers.Dense(output_unit,activation='sigmoid'))

    def call(self, inputs, *args, **kwargs):
        return 2*self.gate(inputs)


class PosoForMLPLayer(tf.keras.layers.Layer):
    """
    input=tf.random.normal((3,4,5))
    layer=PosoForMLPLayer()
    print(layer(input,input))
    """
    def __init__(self,units=[32,16,8],activations=['relu','relu','relu'],gate_hidden_multiple=2,norm=None):
        super().__init__()
        self.dense_layers=[tf.keras.layers.Dense(unit) for unit in units]
        self.activations=[tf.keras.layers.Activation(activation) if activation!='PReLU' else tf.keras.layers.PReLU() for activation in activations]
        self.gates=[GateNULayer(gate_hidden_multiple*unit,unit) for unit in units]
        self.layer_depth=len(units)
        self.norm=norm
        self.C_list=[tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32) for _ in range(len(units))]

    def call(self, inputs_for_main,inputs_for_gate, *args, **kwargs):
        for i in range(self.layer_depth):
            dense_output=self.dense_layers[i](inputs_for_main)
            gate_output=self.gates[i](inputs_for_gate)
            layer_output=tf.multiply(dense_output,gate_output)
            layer_output=self.C_list[i]*layer_output
            if self.norm:
                if self.norm=='batchnorm':
                    layer_output=tf.keras.layers.BatchNormalization()(layer_output)
                if self.norm=='layernorm':
                    layer_output = tf.keras.layers.LayerNormalization()(layer_output)
            inputs=self.activations[i](layer_output)
        return inputs


class MHA(tf.keras.layers.Layer):
    def __init__(self,num_heads,d_model):
        super().__init__()
        self.head_dim=d_model//num_heads
        assert d_model%num_heads==0

        self.num_heads=num_heads
        self.d_model=d_model

        self.wq=tf.keras.layers.Dense(d_model)
        self.wk=tf.keras.layers.Dense(d_model)
        self.wv=tf.keras.layers.Dense(d_model)

        self.wo=tf.keras.layers.Dense(d_model)

    def split_heads(self,input):
        batch_size=tf.shape(input)[0]
        sequence_length=tf.shape(input)[1]
        output=tf.reshape(input,[batch_size,sequence_length,self.num_heads,self.head_dim])
        output=tf.transpose(output,[0,2,1,3])  # (batch_size,num_heads,sequence_length,head_dim)
        return output

    def call(self, q,k,v,mask=None):
        q,k,v=self.wq(q),self.wk(k),self.wv(v)
        q,k,v=self.split_heads(q),self.split_heads(k),self.split_heads(v)
        attention=tf.matmul(q,k,transpose_b=True)
        attention=attention/tf.math.sqrt(tf.cast(self.head_dim,dtype=tf.float32))

        if mask is not None:
            mask=tf.expand_dims(tf.cast(mask,dtype=tf.float32),axis=1)
            attention+=mask*(-1e9)
        attention=tf.nn.softmax(attention,axis=-1)
        output=tf.matmul(attention,v)
        output=tf.transpose(output,[0,2,1,3])
        output=tf.reshape(output,[tf.shape(output)[0],tf.shape(output)[1],-1])
        output=self.wo(output)
        return output


class PosoForMHALayer(tf.keras.layers.Layer):
    """
    temp_mha = PosoForMHALayer(d_model=512, num_heads=8)
    y = tf.random.uniform((3, 60, 512))  # 示例输入 (batch_size, seq_len, d_model)
    inputs_for_gate=tf.random.normal((3,20))
    random_mask = tf.random.uniform((3,60,60), minval=0, maxval=2, dtype=tf.int32)
    random_mask = tf.cast(random_mask, dtype=tf.float32)
    out = temp_mha(inputs_for_gate,y, y, y, mask=random_mask)
    print(out.shape)  # 输出应为 (batch_size, seq_len, d_model)
    """
    def __init__(self,num_heads=8,d_model=512,n_v=6,gate_k_hidden_unit=128,gate_v_hidden_unit=32):
        super().__init__()
        self.head_dim=d_model//num_heads
        assert d_model%num_heads==0

        self.num_heads=num_heads
        self.d_model=d_model
        self.n_v=n_v

        self.wq=tf.keras.layers.Dense(d_model)
        self.wk=tf.keras.layers.Dense(d_model)
        self.wv_list=[tf.keras.layers.Dense(d_model) for _ in range(n_v)]
        self.wo=tf.keras.layers.Dense(d_model)

        self.gate_for_k=GateNULayer(gate_k_hidden_unit,output_unit=d_model)
        self.gate_for_v=GateNULayer(gate_v_hidden_unit,output_unit=n_v)

        self.C = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)

    def split_heads(self,input):
        batch_size=tf.shape(input)[0]
        sequence_length=tf.shape(input)[1]
        output=tf.reshape(input,[batch_size,sequence_length,self.num_heads,self.head_dim])
        output=tf.transpose(output,[0,2,1,3])  # (batch_size,num_heads,sequence_length,head_dim)
        return output

    def call(self, inputs_for_gate,q,k,v,mask=None):
        q,k=self.wq(q),self.wk(k)
        v=tf.stack([wv(v) for wv in self.wv_list],axis=1) # (batch_size,n_v,sequence_length,d_model)

        gate_k=self.gate_for_k(inputs_for_gate)   # (batch_size,d_model)
        gate_v=self.gate_for_v(inputs_for_gate)   # (batch_size,n_v)

        k=tf.multiply(k,tf.expand_dims(gate_k,axis=1))  # (batch_size,sequence_length,d_model)
        v=tf.reduce_sum(tf.multiply(v,tf.expand_dims(tf.expand_dims(gate_v,-1),-1)),axis=1)  # (batch_size,sequence_length,d_model)

        # 普通MHA的逻辑
        q,k,v=self.split_heads(q),self.split_heads(k),self.split_heads(v)
        attention=tf.matmul(q,k,transpose_b=True)
        attention=attention/tf.math.sqrt(tf.cast(self.head_dim,dtype=tf.float32))

        if mask is not None:
            mask=tf.expand_dims(tf.cast(mask,dtype=tf.float32),axis=1)
            attention+=mask*(-1e9)
        attention=tf.nn.softmax(attention,axis=-1)
        output=tf.matmul(attention,v)
        output = self.C * output
        output=tf.transpose(output,[0,2,1,3])
        output=tf.reshape(output,[tf.shape(output)[0],tf.shape(output)[1],-1])
        output=self.wo(output)
        return output


class PosoForMMOELayer(tf.keras.layers.Layer):
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
              'click_goods_num_origin': tf.constant([0.2, 7.8, 4.9]),
              'click_goods_num_square': tf.constant([5.3, 1.2, 8.0]),
              'click_goods_num_cube': tf.constant([-3.8, -19.6, 4.2]),
              'poso_gate_cate1':tf.constant([27,28,29]),
              'poso_gate_cate2':tf.constant([30,31,32]),
              'poso_gate_cont1':tf.constant([0.2, 7.8, 4.9]),
              'poso_gate_cont2':tf.constant([5.3, 1.2, 8.0])}
    layer = PosoForMMOELayer()
    set_custom_initialization(layer)
    print(layer(inputs))
    print('****************')
    input_dict = {}
    max_length = 10
    neg_samples = 6
    for feature in layer.categorical_features+layer.inputs_for_poso_gate_cate_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.continuous_features+layer.inputs_for_poso_gate_cont_features:
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
                 inputs_for_poso_gate_cate_features=['poso_gate_cate1','poso_gate_cate2'],
                 inputs_for_poso_gate_cont_features=['poso_gate_cont1','poso_gate_cont2'],
                 feature_dims=160000,embedding_dims=8, expert_num=3,expert_model_units=[64,8],gate_units=[64],
                 output_units=[64,8],expert_activation='ReLU',gate_activation='ReLU',output_activation='ReLU',
                 gate_hidden_dim=16):
        super().__init__()
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.inputs_for_poso_gate_cate_features=inputs_for_poso_gate_cate_features
        self.inputs_for_poso_gate_cont_features=inputs_for_poso_gate_cont_features
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims, embedding_dims)
        self.expert_num=expert_num

        # expert/gate/output_model都可以替换成其它结构，只要保证输入/输出单元数一致即可
        # expert_model:MLP
        self.expert_model=[make_mlp_layer(units=expert_model_units,activation=expert_activation) for _ in range(self.expert_num)]
        self.poso_gate=GateNULayer(gate_hidden_dim,expert_num)

        # gate_model:MLP
        self.ctr_gate=make_mlp_layer(units=gate_units+[expert_num],activation=gate_activation)
        self.cvr_gate=make_mlp_layer(units=gate_units+[expert_num],activation=gate_activation)

        # output_model:MLP
        self.ctr_output=make_mlp_layer(units=output_units,activation=output_activation,sigmoid_units=True)
        self.cvr_output=make_mlp_layer(units=output_units,activation=output_activation,sigmoid_units=True)

        self.C = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)

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
        gate=tf.nn.softmax(gate,axis=-1)
        gate = tf.expand_dims(gate, axis=2)
        input = tf.multiply(expert_outputs, gate)
        input = tf.keras.layers.Flatten()(input)
        output = output(input)
        output = self.C*output
        return output

    def generate_poso_gate_output(self,inputs):
        X_gate_cate = []
        for feature in self.inputs_for_poso_gate_cate_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_gate_cate.append(feature_tensor)
        X_gate_cate = tf.concat(X_gate_cate, axis=1)

        emb_output = self.embedding_layer(X_gate_cate)
        emb_output = tf.keras.layers.Flatten()(emb_output)

        X_gate_cont = []
        for feature in self.inputs_for_poso_gate_cont_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_gate_cont.append(feature_tensor)
        X_gate_cont = tf.concat(X_gate_cont, axis=1)

        X_gate_combined = tf.concat([emb_output, X_gate_cont], axis=1)

        poso_gate_output=self.poso_gate(X_gate_combined)
        return poso_gate_output

    def call(self, inputs):
        X_combined,expert_outputs=self.generate_expert_output(inputs)
        poso_gate_output=self.generate_poso_gate_output(inputs)
        poso_gated_expert_output=tf.expand_dims(poso_gate_output,axis=-1)*expert_outputs
        ctr_output=self.generate_task_output(X_combined,poso_gated_expert_output,task='ctr')
        cvr_output = self.generate_task_output(X_combined, poso_gated_expert_output, task='cvr')
        result={'ctr_output':ctr_output,'cvr_output':cvr_output}
        return result


class PEPNetLayer(tf.keras.layers.Layer):
    """
    inputs = {'sdk_type': tf.constant([0, 1, 2]), 'remote_host': tf.constant([3, 4, 5]),
              'device_type': tf.constant([6, 7, 8]), 'dtu': tf.constant([9, 10, 11]),
              'click_goods_num': tf.constant([12, 13, 14]), 'buy_click_num': tf.constant([15, 16, 17]),
              'goods_show_num': tf.constant([18, 19, 20]), 'goods_click_num': tf.constant([21, 22, 23]),
              'brand_name': tf.constant([24, 25, 26]),
              'ppnet_cate1': tf.constant([27, 28, 29]),
              'ppnet_cate2': tf.constant([30, 31, 32]),
              'epnet_cate1': tf.constant([33, 34, 35]),
              'epnet_cate2': tf.constant([36, 37, 38])}
    layer = PEPNetLayer()
    set_custom_initialization(layer)
    print(layer(inputs))
    print('****************')
    input_dict = {}
    max_length = 10
    neg_samples = 6
    for feature in layer.categorical_features + layer.ppnet_input_features+layer.epnet_input_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    output = layer(input_dict)
    model = tf.keras.Model(input_dict, output)
    model.summary()
    print(model(inputs))
    """
    def __init__(self,
                 categorical_features=['sdk_type', 'remote_host', 'device_type', 'dtu',
                                       'click_goods_num', 'buy_click_num', 'goods_show_num', 'goods_click_num',
                                       'brand_name'],
                 ppnet_input_features=['ppnet_cate1','ppnet_cate2'],
                 epnet_input_features=['epnet_cate1','epnet_cate2'],
                 feature_dims=160000,embedding_dims=8,
                 gate_hidden_dim=16,num_tasks=3,poso_mlp_units=[64,16,1]):
        super().__init__()
        self.categorical_features = categorical_features
        self.ppnet_input_features=ppnet_input_features
        self.epnet_input_features=epnet_input_features
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims, embedding_dims)

        # 论文中所有embedding共享一个poso_gate，也可以设置成独立，形成列表
        self.poso_gate_for_embedding=[GateNULayer(gate_hidden_dim,embedding_dims) for _ in range(len(self.categorical_features))]
        # 论文中所有task共享一个poso_gate，也可以设置成独立，形成列表
        self.poso_mlp_layers=[PosoForMLPLayer(units=poso_mlp_units,activations=['relu','relu','sigmoid'],gate_hidden_multiple=2,norm=None) for _ in range(num_tasks)]

    def stack_input_features(self,inputs):
        X_cate = []
        for feature in self.categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            X_cate.append(feature_tensor)
        X_cate = tf.concat(X_cate, axis=1)
        # (batch_size, main_features, embedding_dim)
        main_input = self.embedding_layer(X_cate)

        ppnet_cate = []
        for feature in self.ppnet_input_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            ppnet_cate.append(feature_tensor)
        ppnet_cate = tf.concat(ppnet_cate, axis=1)

        ppnet_input = self.embedding_layer(ppnet_cate)
        ppnet_input = tf.keras.layers.Flatten()(ppnet_input)

        # 论文中epnet的输入也拼接模型的主输入特征，这里去除主特征，只保留场景强相关特征
        epnet_cate = []
        for feature in self.ppnet_input_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            epnet_cate.append(feature_tensor)
        epnet_cate = tf.concat(epnet_cate, axis=1)

        epnet_input = self.embedding_layer(epnet_cate)
        epnet_input = tf.keras.layers.Flatten()(epnet_input)

        # 论文中连续特征也离散化并取embedding，所以认为连续特征已处理并合并到离散特征中
        return main_input,ppnet_input,epnet_input

    def call(self, inputs, *args, **kwargs):
        main_input,ppnet_input,epnet_input=self.stack_input_features(inputs)
        epnet_output=[gate(epnet_input) for gate in self.poso_gate_for_embedding]
        epnet_output=tf.stack(epnet_output,axis=1) # (batch_size,main_features,embedding_dim)
        posoed_embedding=tf.multiply(epnet_output,main_input)
        concated_embedding=tf.keras.layers.Flatten()(posoed_embedding)
        main_for_ppnet=tf.stop_gradient(concated_embedding)
        ppnet_input=tf.concat([main_for_ppnet,ppnet_input],axis=1)
        task_outputs=[gate(concated_embedding,ppnet_input) for gate in self.poso_mlp_layers]
        task_outputs=tf.stack(task_outputs,axis=1)
        return task_outputs


if __name__ == '__main__':
    inputs = {'sdk_type': tf.constant([0, 1, 2]), 'remote_host': tf.constant([3, 4, 5]),
              'device_type': tf.constant([6, 7, 8]), 'dtu': tf.constant([9, 10, 11]),
              'click_goods_num': tf.constant([12, 13, 14]), 'buy_click_num': tf.constant([15, 16, 17]),
              'goods_show_num': tf.constant([18, 19, 20]), 'goods_click_num': tf.constant([21, 22, 23]),
              'brand_name': tf.constant([24, 25, 26]),
              'ppnet_cate1': tf.constant([27, 28, 29]),
              'ppnet_cate2': tf.constant([30, 31, 32]),
              'epnet_cate1': tf.constant([33, 34, 35]),
              'epnet_cate2': tf.constant([36, 37, 38])}
    layer = PEPNetLayer()
    set_custom_initialization(layer)
    print(layer(inputs))
    print('****************')
    input_dict = {}
    max_length = 10
    neg_samples = 6
    for feature in layer.categorical_features + layer.ppnet_input_features + layer.epnet_input_features:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    output = layer(input_dict)
    model = tf.keras.Model(input_dict, output)
    model.summary()
    print(model(inputs))

