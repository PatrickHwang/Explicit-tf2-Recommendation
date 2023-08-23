import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from Tools import CustomTFWriter

class DataGenerator:
    def __init__(self,train_path='data/raw/170',test_path='data/raw/180',output_path='data/generated',
                 user_features=['sdk_type', 'remote_host', 'device_type', 'dtu',
    'click_goods_num', 'buy_click_num'],item_features=['goods_show_num', 'goods_click_num',
    'brand_name'],label_names=['ctr','cvr']):
        self.train_path=train_path
        self.test_path=test_path
        self.output_path=output_path
        self.user_features=user_features
        self.item_features=item_features
        self.label_names=label_names
        self.encode_columns=user_features+item_features
        self.recorder = {}

    def gen_dataframe(self,mode='train'):
        total=[]
        if mode=='train':
            path=self.train_path
        else:
            path=self.test_path
        with open(path,'r') as f:
            lines=f.readlines()
            '''
            {'UserId': 'DD09A08B-EFC8-47E3-9E99-40678978BC00', 'ItemId': '70846665513', 'Label': '0$#0', 
            'UserFeature': {'UserId': 'DD09A08B-EFC8-47E3-9E99-40678978BC00', 
                            'Feature': [{'FeatureName': 'device_id', 'FeatureValue': {'StringValue': ''}}, 
                                        {'FeatureName': 'user_id', 'FeatureValue': {'StringValue': '1022020cdec4bcf7c83cc99b020e6d9f4b0bb3'}}, 
                                        {'FeatureName': 'sdk_type', 'FeatureValue': {'StringValue': 'ios'}}, 
                                        {'FeatureName': 'remote_host', 'FeatureValue': {'StringValue': '117.39.233.220'}}, 
                                        {'FeatureName': 'device_brand', 'FeatureValue': {'StringValue': 'iPhone'}}, 
                                        {'FeatureName': 'device_type', 'FeatureValue': {'StringValue': 'iPhone SE 2'}}, 
                                        {'FeatureName': 'dtu', 'FeatureValue': {'StringValue': 'ios01'}}, 
                                        {'FeatureName': 'click_goods_num', 'FeatureValue': {'FloatValue': 0}}, 
                                        {'FeatureName': 'click_goods', 'FeatureValue': {'StringValue': ''}}, 
                                        {'FeatureName': 'click_goods_list', 'FeatureValue': {}}, 
                                        {'FeatureName': 'agm_page_num', 'FeatureValue': {'FloatValue': 7}}, 
                                        {'FeatureName': 'detail_page_num', 'FeatureValue': {'FloatValue': 1}}, 
                                        {'FeatureName': 'share_num', 'FeatureValue': {'FloatValue': 0}}, 
                                        {'FeatureName': 'share_goods', 'FeatureValue': {'StringValue': ''}}, 
                                        {'FeatureName': 'share_goods_list', 'FeatureValue': {}}, 
                                        {'FeatureName': 'buy_click_num', 'FeatureValue': {'FloatValue': 0}}, 
                                        {'FeatureName': 'buy_click_goods', 'FeatureValue': {'StringValue': ''}}, 
                                        {'FeatureName': 'buy_click_goods_list', 'FeatureValue': {}}]}, 
            'ItemFeature': {'ItemId': '70846665513', 
                            'Feature': [{'FeatureName': 'goods_buy_num', 'FeatureValue': {'FloatValue': 0}}, 
                                        {'FeatureName': 'goods_click_num', 'FeatureValue': {'FloatValue': 1}}, 
                                        {'FeatureName': 'goods_share_num', 'FeatureValue': {'FloatValue': 0}}, 
                                        {'FeatureName': 'goods_show_num', 'FeatureValue': {'FloatValue': 7544}}, 
                                        {'FeatureName': 'brand_name', 'FeatureValue': {'StringValue': ''}}, 
                                        {'FeatureName': 'cate1', 'FeatureValue': {'StringValue': ''}}, 
                                        {'FeatureName': 'cate2', 'FeatureValue': {'StringValue': ''}}, 
                                        {'FeatureName': 'cate3', 'FeatureValue': {'StringValue': ''}}, 
                                        {'FeatureName': 'commission', 'FeatureValue': {'FloatValue': 0}}, 
                                        {'FeatureName': 'goods_id', 'FeatureValue': {'StringValue': ''}}, 
                                        {'FeatureName': 'ist_date', 'FeatureValue': {'Int32Value': 0}}, 
                                        {'FeatureName': 'lowest_price', 'FeatureValue': {'FloatValue': 0}}, 
                                        {'FeatureName': 'price', 'FeatureValue': {'FloatValue': 0}}, 
                                        {'FeatureName': 'site_id', 'FeatureValue': {'Int32Value': 0}}, 
                                        {'FeatureName': 'source', 'FeatureValue': {'Int32Value': 0}}, 
                                        {'FeatureName': 'status', 'FeatureValue': {'Int32Value': 1}}]}}
            '''
            for line in lines:
                record=json.loads(line)
                result=[]
                user_id=record['UserId']
                item_id=record['ItemId']
                user_feature=record["UserFeature"].get("Feature", None)
                item_feature = record["ItemFeature"].get("Feature", None)
                labels=record['Label'].split('$#')
                result.append(user_id)
                result.append(item_id)
                for feature in self.user_features:
                    if not user_feature:
                        continue
                    for row in user_feature:
                        if row['FeatureName']==feature:
                            result.append(list(row['FeatureValue'].values())[0])
                for feature in self.item_features:
                    if not item_feature:
                        continue
                    for row in item_feature:
                        if row['FeatureName']==feature:
                            result.append(list(row['FeatureValue'].values())[0])
                result.extend(labels)
                total.append(result)
        total=pd.DataFrame(total,columns=['UserId','ItemId']+self.user_features+self.item_features+self.label_names)
        if mode=='train':
            total['mode']='train'
        else:
            total['mode']='test'
        print(total.shape)
        return total

    def get_feature_dims(self):
        df_train=self.gen_dataframe('train')
        df_test=self.gen_dataframe('test')
        df_all=pd.concat([df_train,df_test],axis=0)
        '''
        buy_click_num      348
        goods_show_num     348
        goods_click_num    348
        brand_name         450
        ctr                450
        cvr                450
        '''
        df_all=df_all[(df_all['ctr'].notnull()) & (df_all['cvr'].notnull())].fillna(0)
        df_all['click_goods_num_origin']=df_all['click_goods_num']
        df_all['click_goods_num_square'] =df_all['click_goods_num']**2
        df_all['click_goods_num_cube'] = df_all['click_goods_num'] ** 3
        self.df_all=df_all
        # print(df_all[['ctr','cvr']].value_counts())
        self.feature_dims = [len(self.df_all[col].unique()) for col in self.encode_columns]
        self.feature_offsets = (0, *np.cumsum(self.feature_dims[:-1]))
        self.feature_values_cnt = sum(self.feature_dims)

    def encode_and_record(self):
        # 独热处理，记得加上特征偏移量
        le = LabelEncoder()
        for index, col in enumerate(self.encode_columns):
            offset = self.feature_offsets[index]
            self.df_all[col] = le.fit_transform(self.df_all[col]) + offset
            self.recorder[col] = {str(value): int(i + offset) for i, value in enumerate(le.classes_)}
        # print(self.recorder)
        with open(self.output_path + '/feature_dict.json', 'w') as f:
            json.dump(self.recorder, f)

    def write_tf_records(self):
        # 写入tf数据文件
        counter = {'train': 0, 'test': 0}
        custom_writer = CustomTFWriter(data_name='mmoe', tf_doc_record_limit=200000,
                                       output_path=self.output_path)
        all_cols = self.encode_columns + ['click_goods_num_origin','click_goods_num_square','click_goods_num_cube',
                                          'mode', 'ctr','cvr']
        for line in self.df_all[all_cols].values:
            click_goods_num_origin,click_goods_num_square,click_goods_num_cube,data_type, ctr,cvr = line[-6:]
            features = line[:-6]
            sample = {}
            sample["ctr"] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(ctr)]))
            sample["cvr"] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(cvr)]))
            sample['click_goods_num_origin'] = tf.train.Feature(float_list=tf.train.FloatList(
                value=[click_goods_num_origin]))
            sample['click_goods_num_square'] = tf.train.Feature(float_list=tf.train.FloatList(
                value=[click_goods_num_square]))
            sample['click_goods_num_cube'] = tf.train.Feature(float_list=tf.train.FloatList(
                value=[click_goods_num_cube]))
            for i, feature in enumerate(self.encode_columns):
                sample[feature] = tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[features[i]]))
            sample = tf.train.Example(features=tf.train.Features(feature=sample))
            if data_type == 'train':
                custom_writer.write(sample, 'train')
            elif data_type == 'test':
                custom_writer.write(sample, 'test')
            counter[data_type] += 1
        print(counter)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    dg=DataGenerator()
    dg.get_feature_dims()
    dg.encode_and_record()
    dg.write_tf_records()