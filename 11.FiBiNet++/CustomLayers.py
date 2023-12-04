import tensorflow as tf
from tensorflow.python.ops.math_ops import MatMul
from tensorflow.python.ops.nn_ops import BiasAdd
from tensorflow.python.keras import activations, initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D
from tensorflow.keras.layers import Embedding,BatchNormalization,LayerNormalization,MultiHeadAttention,Dropout,Lambda,Activation
from tensorflow.keras.regularizers import l2
import itertools

from tensorflow.python.ops.init_ops_v2 import glorot_normal
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
        elif activation!="None":
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


class NormInputFeaturesEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self,categorical_features=['uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'itag1', 'itag2', 'itag3',
                                       'itag4'],
                 continuous_features=['itag4_origin', 'itag4_square', 'itag4_cube'], feature_dims=160000,
                 embedding_dims=16):
        super(NormInputFeaturesEmbeddingLayer, self).__init__()
        self.categorical_features = categorical_features
        self.continuous_features_keys = [name + '_key' for name in continuous_features]
        self.continuous_features_values = [name + '_value' for name in continuous_features]
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims, embedding_dims)
        self.emb_batchnorm = tf.keras.layers.BatchNormalization()
        self.emb_layernorm_list = [tf.keras.layers.LayerNormalization() for _ in range(len(continuous_features))]

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        # 离散特征，embedding过Batchnorm
        cate = []
        for feature in self.categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            cate.append(feature_tensor)
        X_cate = tf.concat(cate, axis=1)

        X_cate_emb = self.embedding_layer(X_cate)
        X_cate_emb_normed=self.emb_batchnorm(X_cate_emb)

        # 连续特征，原始值*embedding并且每个域过一个专属Layernorm
        cont_keys = []
        for feature in self.continuous_features_keys:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            cont_keys.append(feature_tensor)
        X_cont_keys = tf.concat(cont_keys, axis=1)

        cont_values = []
        for feature in self.continuous_features_values:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            cont_values.append(feature_tensor)
        X_cont_values = tf.concat(cont_values, axis=1)

        X_cont_emb = self.embedding_layer(X_cont_keys)

        X_cont_emb=X_cont_emb*tf.expand_dims(X_cont_values,axis=-1)

        output_tensors = []

        # 对每个field应用Layer Normalization
        for i,layernorm in enumerate(self.emb_layernorm_list):
            # 提取第 i 个field
            field_tensor = X_cont_emb[:, i, :]

            # 对该field应用Layer Normalization
            norm_field_tensor = layernorm(field_tensor)

            # 将处理后的field添加到列表中
            output_tensors.append(norm_field_tensor)

        # 将所有处理后的fields沿着第二维度拼接起来
        X_cont_emb_normed = tf.stack(output_tensors, axis=1)

        X_input=tf.concat([X_cate_emb_normed,X_cont_emb_normed],axis=1)
        return X_input


class FiBiNetPlusLayer(tf.keras.layers.Layer):
    """
        for feature in layer.norm_embedding_layer.categorical_features+layer.norm_embedding_layer.continuous_features_keys:
            input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
        for feature in layer.norm_embedding_layer.continuous_features_values:
            input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
    """
    def __init__(self,
                 categorical_features=['uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'itag1', 'itag2', 'itag3',
                                       'itag4'],
                 continuous_features=['itag4_origin', 'itag4_square', 'itag4_cube'], feature_dims=160000,
                 embedding_dims=16,bilinear_type='interaction',bilinear_output_dim=16,senet_reduction_ratio=3,senet_group_num=2,
                 final_mlp_units=[32],final_mlp_activation='ReLU'):
        super(FiBiNetPlusLayer, self).__init__()
        self.norm_embedding_layer=NormInputFeaturesEmbeddingLayer(categorical_features,continuous_features,feature_dims,embedding_dims)
        self.bilinear_interaction_plus_layer=BilinearInteractionPlusLayer(bilinear_type,bilinear_output_dim)
        self.senet_plus_layer=SENetPlusLayer(senet_reduction_ratio,senet_group_num)
        self.final_mlp=make_mlp_layer(final_mlp_units,activation=final_mlp_activation,sigmoid_units=True)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        X_input=self.norm_embedding_layer(inputs)
        X_bilinear_output=self.bilinear_interaction_plus_layer(X_input)
        X_senet_output=self.senet_plus_layer(X_input)
        X_senet_output=tf.reshape(X_senet_output,[-1,tf.shape(X_senet_output)[1]*tf.shape(X_senet_output)[2]])
        X_extracted=tf.concat([X_bilinear_output,X_senet_output],axis=1)
        logits=self.final_mlp(X_extracted)
        result = {'output': logits}
        return result


class SENetPlusLayer(tf.keras.layers.Layer):
    def __init__(self,reduction_ratio=3,group_num=4):
        super(SENetPlusLayer,self).__init__()
        self.reduction_ratio=reduction_ratio
        self.group_num=group_num

    def build(self,input_shape):
        self.field_num=input_shape[1]
        self.embedding_size=input_shape[2]

        self.mid_unit_num=max(1,2*self.group_num*self.field_num//self.reduction_ratio)
        self.excitation=make_mlp_layer([self.mid_unit_num,self.field_num*self.embedding_size],activation='relu')
        super(SENetPlusLayer, self).build(input_shape)

    def call(self,inputs):
        regrouped_inputs=tf.split(inputs,self.group_num,axis=2)
        regrouped_inputs = tf.stack(regrouped_inputs,axis=2)
        grouped_max=tf.reduce_max(regrouped_inputs,axis=-1)
        grouped_mean=tf.reduce_mean(regrouped_inputs,axis=-1)
        grouped_info=tf.concat([grouped_mean,grouped_max],axis=-1)
        grouped_info=tf.reshape(grouped_info,[-1,tf.shape(grouped_info)[1]*tf.shape(grouped_info)[2]])
        A=self.excitation(grouped_info)
        A=tf.reshape(A,[-1,tf.shape(inputs)[1],tf.shape(inputs)[2]])
        V=tf.multiply(inputs,A)
        return V


class BilinearInteractionPlusLayer(tf.keras.layers.Layer):
    def __init__(self,bilinear_type='interaction',output_dim=16):
        super(BilinearInteractionPlusLayer,self).__init__()
        self.bilinear_type=bilinear_type
        self.reducing_layer=make_mlp_layer([output_dim],activation='None')

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
        super(BilinearInteractionPlusLayer, self).build(input_shape)

    def call(self,inputs):
        # (None,field_num,embedding_size) -> [n*(None,1,embedding_size)]
        field_list=tf.split(inputs,self.field_num,axis=1)
        field_list = [tf.squeeze(field, axis=1) for field in field_list]
        if self.bilinear_type=='all':
            p=[tf.einsum('be,be->b',tf.tensordot(field_list[i],self.W,axes=(-1,0)),field_list[j]) for i,j in itertools.combinations(range(self.field_num), 2)]
        elif self.bilinear_type=='each':
            p=[tf.einsum('be,be->b',tf.tensordot(field_list[i],self.W_list[i],axes=(-1,0)),field_list[j]) for i,j in itertools.combinations(range(self.field_num), 2)]
        elif self.bilinear_type=='interaction':
            p=[tf.einsum('be,be->b',tf.tensordot(field_list[v[0]],w,axes=(-1,0)),field_list[v[1]]) for v,w in zip(itertools.combinations(range(self.field_num), 2),self.W_list)]
        else:
            raise NotImplementedError
        # (None,fields_interaction_num)
        output=tf.stack(p,axis=1)
        output=self.reducing_layer(output)
        return output


class LayerNormInputFeaturesEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self,categorical_features=['uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'itag1', 'itag2', 'itag3',
                                       'itag4'],
                 continuous_features=['itag4_origin', 'itag4_square', 'itag4_cube'], feature_dims=160000,
                 embedding_dims=16):
        super(LayerNormInputFeaturesEmbeddingLayer, self).__init__()
        self.categorical_features = categorical_features
        self.continuous_features_keys = [name + '_key' for name in continuous_features]
        self.continuous_features_values = [name + '_value' for name in continuous_features]
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims, embedding_dims)
        self.emb_layernorm_list = [tf.keras.layers.LayerNormalization() for _ in range(len(categorical_features)+len(continuous_features))]

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        # 离散特征，embedding过Batchnorm
        cate = []
        for feature in self.categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            cate.append(feature_tensor)
        X_cate = tf.concat(cate, axis=1)

        X_cate_emb = self.embedding_layer(X_cate)

        # 连续特征，原始值*embedding并且每个域过一个专属Layernorm
        cont_keys = []
        for feature in self.continuous_features_keys:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            cont_keys.append(feature_tensor)
        X_cont_keys = tf.concat(cont_keys, axis=1)

        cont_values = []
        for feature in self.continuous_features_values:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            cont_values.append(feature_tensor)
        X_cont_values = tf.concat(cont_values, axis=1)

        X_cont_emb = self.embedding_layer(X_cont_keys)

        X_cont_emb=X_cont_emb*tf.expand_dims(X_cont_values,axis=-1)

        X_emb=tf.concat([X_cate_emb,X_cont_emb],axis=1)

        output_tensors = []

        # 对每个field应用Layer Normalization
        for i,layernorm in enumerate(self.emb_layernorm_list):
            # 提取第 i 个field
            field_tensor = X_emb[:, i, :]

            # 对该field应用Layer Normalization
            norm_field_tensor = layernorm(field_tensor)

            # 将处理后的field添加到列表中
            output_tensors.append(norm_field_tensor)

        # 将所有处理后的fields沿着第二维度拼接起来
        X_emb_normed = tf.stack(output_tensors, axis=1)

        return X_emb_normed,X_emb


def make_instance_guided_mask(output_dim,reduction_rate=3):
    mlp = tf.keras.models.Sequential()
    mlp.add(tf.keras.layers.Dense(output_dim*reduction_rate))
    mlp.add(tf.keras.layers.ReLU())
    mlp.add(tf.keras.layers.Dense(output_dim))
    return mlp


class MaskBlockLayer(tf.keras.layers.Layer):
    def __init__(self,fields_num=13,input_type='feature',embedding_dims=16,block_output_dim=32):
        super(MaskBlockLayer, self).__init__()
        guided_input_dim=fields_num*embedding_dims if input_type=='feature' else block_output_dim
        self.instance_guided_mask=make_instance_guided_mask(guided_input_dim)
        self.ln_hid=make_mlp_layer([block_output_dim],activation='ReLU',normalization='layernorm')

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        X_emb_normed, X_emb=inputs
        guided_mask=self.instance_guided_mask(X_emb)
        hidden_vector=tf.multiply(X_emb_normed,guided_mask)
        output_vector=self.ln_hid(hidden_vector)
        return output_vector


class SerialMaskNetLayer(tf.keras.layers.Layer):
    def __init__(self,
                 categorical_features=['uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'itag1', 'itag2', 'itag3',
                                       'itag4'],
                 continuous_features=['itag4_origin', 'itag4_square', 'itag4_cube'],feature_dims=160000,
                 embedding_dims=16,block_output_dim=32,block_num=6):
        super(SerialMaskNetLayer, self).__init__()
        self.norm_embedding_layer=LayerNormInputFeaturesEmbeddingLayer(categorical_features,continuous_features,feature_dims,embedding_dims)
        self.embedding_dims=embedding_dims
        self.fields_num=len(categorical_features)+len(continuous_features)
        self.mask_block_on_feature=MaskBlockLayer(fields_num=self.fields_num,input_type='feature',embedding_dims=embedding_dims,block_output_dim=block_output_dim)
        self.mask_block_on_block_list=[MaskBlockLayer(fields_num=self.fields_num,input_type='block',embedding_dims=embedding_dims,block_output_dim=block_output_dim) for _ in range(block_num-1)]


    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        X_emb_normed,X_emb=self.norm_embedding_layer(inputs)
        X_emb_normed=tf.reshape(X_emb_normed,[-1,self.embedding_dims*self.fields_num])
        X_emb = tf.reshape(X_emb, [-1, self.embedding_dims * self.fields_num])
        X = self.mask_block_on_feature((X_emb_normed,X_emb))
        for block in self.mask_block_on_block_list:
            X = block((X,X_emb))
        return X


class ParralledMaskNetLayer(tf.keras.layers.Layer):
    def __init__(self,
                 categorical_features=['uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'itag1', 'itag2', 'itag3',
                                       'itag4'],
                 continuous_features=['itag4_origin', 'itag4_square', 'itag4_cube'],feature_dims=160000,
                 embedding_dims=16,block_output_dim=32,block_num=6):
        super(ParralledMaskNetLayer, self).__init__()
        self.norm_embedding_layer=LayerNormInputFeaturesEmbeddingLayer(categorical_features,continuous_features,feature_dims,embedding_dims)
        self.fields_num=len(categorical_features)+len(continuous_features)
        self.mask_block_on_feature_list=[MaskBlockLayer(fields_num=self.fields_num,input_type='feature',embedding_dims=embedding_dims,block_output_dim=block_output_dim) for _ in range(block_num)]

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        X_emb_normed,X_emb=self.norm_embedding_layer(inputs)
        X_list=[block((X_emb_normed,X_emb)) for block in self.mask_block_on_feature_list]
        X=tf.concat(X_list,axis=1)
        return X


class MaskNetLayer(tf.keras.layers.Layer):
    def __init__(self,
                 categorical_features=['uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'itag1', 'itag2', 'itag3',
                                       'itag4'],
                 continuous_features=['itag4_origin', 'itag4_square', 'itag4_cube'],feature_dims=160000,
                 embedding_dims=16,block_output_dim=32,block_num=6,stacking_mode='serial',final_mlp_units=[32]):
        super(MaskNetLayer, self).__init__()
        self.categorical_features=categorical_features
        self.continuous_features_keys = [name + '_key' for name in continuous_features]
        self.continuous_features_values = [name + '_value' for name in continuous_features]
        self.mask_net=SerialMaskNetLayer(categorical_features,continuous_features,feature_dims,embedding_dims,block_output_dim,block_num) if \
            stacking_mode=='serial' else ParralledMaskNetLayer(categorical_features,continuous_features,feature_dims,embedding_dims,block_output_dim,block_num)
        self.final_mlp=make_mlp_layer(final_mlp_units,sigmoid_units=True,normalization='None')

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        mask_block_output=self.mask_net(inputs)
        logits=self.final_mlp(mask_block_output)
        result = {'output': logits}
        return result


class ContextualEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self,fields_num=13,embedding_dims=16):
        super(ContextualEmbeddingLayer, self).__init__()
        self.contextual_embedding_transform=make_instance_guided_mask(fields_num*embedding_dims)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        org_shape=tf.shape(inputs)
        concated_fields=tf.reshape(inputs,[-1,org_shape[1]*org_shape[2]])
        mask=self.contextual_embedding_transform(concated_fields)
        mask=tf.reshape(mask,org_shape)
        return mask


class NonLinearFeedforwardLayer(tf.keras.layers.Layer):
    def __init__(self,embedding_dims=16,mode='pointwise'):
        super(NonLinearFeedforwardLayer, self).__init__()
        self.W1 = self.add_weight(shape=(embedding_dims, embedding_dims),initializer=glorot_normal())
        self.ln = tf.keras.layers.LayerNormalization()
        self.mode=mode
        if mode=='pointwise':
            self.W2 = self.add_weight(shape=(embedding_dims, embedding_dims), initializer=glorot_normal())

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        output=tf.matmul(inputs,self.W1)
        if self.mode=='pointwise':
            output = tf.keras.layers.Activation('ReLU')(output)
            output=tf.matmul(output,self.W2)+inputs
        output=self.ln(output)
        return output


class ContextNetBlockLayer(tf.keras.layers.Layer):
    def __init__(self,nonlinear_type='pointwise'):
        super(ContextNetBlockLayer, self).__init__()
        self.nonlinear_type=nonlinear_type

    def build(self, input_shape):
        self.fields_num=input_shape[1]
        self.embedding_dims=input_shape[2]
        self.nonlinear_layer_list=[NonLinearFeedforwardLayer(embedding_dims=self.embedding_dims,mode=self.nonlinear_type) for _ in range(self.fields_num)]
        self.ce_layer = ContextualEmbeddingLayer(fields_num=self.fields_num,embedding_dims=self.embedding_dims)
        self.built = True

    def call(self, inputs):
        mask=self.ce_layer(inputs)
        field_outputs=[]
        for i in range(self.fields_num):
            field_mask=mask[:,i,:]
            field_input=inputs[:,i,:]
            nonlinear_layer=self.nonlinear_layer_list[i]
            field_output=nonlinear_layer(tf.multiply(field_input,field_mask))
            field_outputs.append(field_output)
        output=tf.stack(field_outputs,axis=1)
        return output


class ContextNetLayer(tf.keras.layers.Layer):
    def __init__(self,
                 categorical_features=['uid', 'iid', 'utag1', 'utag2', 'utag3', 'utag4', 'itag1', 'itag2', 'itag3',
                                       'itag4'],
                 continuous_features=['itag4_origin', 'itag4_square', 'itag4_cube'],feature_dims=160000,
                 embedding_dims=16,block_num=6,final_mlp_units=[32],nonlinear_type='pointwise'):
        super(ContextNetLayer, self).__init__()
        self.categorical_features=categorical_features
        self.continuous_features_keys = [name + '_key' for name in continuous_features]
        self.continuous_features_values = [name + '_value' for name in continuous_features]
        self.embedding_layer = tf.keras.layers.Embedding(feature_dims, embedding_dims)
        self.context_block_list=[ContextNetBlockLayer(nonlinear_type=nonlinear_type) for _ in range(block_num)]
        self.final_mlp=make_mlp_layer(final_mlp_units,sigmoid_units=True,normalization='None')

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        cate = []
        for feature in self.categorical_features:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            cate.append(feature_tensor)
        X_cate = tf.concat(cate, axis=1)

        X_cate_emb = self.embedding_layer(X_cate)

        # 连续特征，原始值*embedding并且每个域过一个专属Layernorm
        cont_keys = []
        for feature in self.continuous_features_keys:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            cont_keys.append(feature_tensor)
        X_cont_keys = tf.concat(cont_keys, axis=1)

        cont_values = []
        for feature in self.continuous_features_values:
            feature_tensor = inputs[feature]
            if len(feature_tensor.shape) == 1:
                feature_tensor = tf.expand_dims(feature_tensor, axis=1)
            cont_values.append(feature_tensor)
        X_cont_values = tf.concat(cont_values, axis=1)

        X_cont_emb = self.embedding_layer(X_cont_keys)

        X_cont_emb = X_cont_emb * tf.expand_dims(X_cont_values, axis=-1)

        X = tf.concat([X_cate_emb, X_cont_emb], axis=1)
        for block in self.context_block_list:
            X=block(X)

        X_flattened=tf.reshape(X,[-1,tf.shape(X)[1]*tf.shape(X)[2]])
        logits=self.final_mlp(X_flattened)

        result = {'output': logits}
        return result


if __name__ == '__main__':
    inputs = {'uid': tf.constant([0, 1, 2]), 'iid': tf.constant([3, 4, 5]),
             'utag1': tf.constant([6, 7, 8]), 'utag2': tf.constant([9, 10, 11]),
             'utag3': tf.constant([12, 13, 14]), 'utag4': tf.constant([15, 16, 17]),
             'itag1': tf.constant([18, 19, 20]), 'itag2': tf.constant([21, 22, 23]),
             'itag3': tf.constant([24, 25, 26]), 'itag4': tf.constant([27, 28, 29]),
             'itag4_origin_key': tf.constant([30, 31, 32]),
             'itag4_square_key': tf.constant([33, 34, 35]),
             'itag4_cube_key': tf.constant([36, 37, 38]),
             'itag4_origin_value': tf.constant([0.2, 7.8, 4.9]),
             'itag4_square_value': tf.constant([5.3, 1.2, 8.0]),
             'itag4_cube_value': tf.constant([-3.8, -19.6, 4.2])}
    layer = ContextNetLayer(nonlinear_type='pointwise')
    set_custom_initialization(layer)
    print(layer(inputs))
    print('****************')
    input_dict = {}
    for feature in layer.categorical_features+layer.continuous_features_keys:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.int64)
    for feature in layer.continuous_features_values:
        input_dict[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)
    output = layer(input_dict)
    model = tf.keras.Model(input_dict, output)
    print(model(inputs))
    model.summary()
