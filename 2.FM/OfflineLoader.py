import json

import argparse
import redis
from sklearn.neighbors import BallTree
import tensorflow as tf
import pickle
import numpy as np

from ModelManager import ModelManager

assert tf.__version__[0] == '2', 'model is built by tensorflow2,please check tf version first.'


class OfflineLoader:
    """和排序模型无关，只需要加载召回模型"""
    def __init__(self, user_profile_path='data/generated/user_profile.json', item_profile_path='data/generated/item_profile.json',
                 ebd_save_path='retrieval_model/ebd_result/',model_path='retrieval_model/output',checkpoint_path='retrieval_model/checkpoint',
                 cluster_center_num=10, fetch_num=20, redis_info={'host': '127.0.0.1', 'port': '6379', 'db': 0},
                 user_feature_names=['user_id','user_tag1','user_tag2'],item_feature_names=['item_id','item_tag1','item_tag2','item_tag3'],
                 layer_name='dssm_double_tower'):
        self.user_profile_path = user_profile_path
        self.item_profile_path = item_profile_path
        self.ebd_save_path = ebd_save_path
        self.model_path=model_path
        self.checkpoint_path=checkpoint_path
        self.cluster_center_num = cluster_center_num
        self.fetch_num = fetch_num
        self.redis_info = redis_info
        self.user_embedding_dict={}
        self.item_embedding_dict={}
        self.redis_client = None
        self.user_feature_names=user_feature_names
        self.item_feature_names=item_feature_names
        self.layer_name=layer_name

    def load_model(self, model_class=ModelManager,source='checkpoint',checkpoint_path=None,model_path=None):
        assert source in ('checkpoint','model')
        self.mm=model_class(layer=self.layer_name)
        if source=='checkpoint':
            if not checkpoint_path:
                checkpoint_path = self.checkpoint_path
            self.mm.load_checkpoint_and_eval(eval=False,checkpoint_path=checkpoint_path)
        else:
            if not model_path:
                model_path = self.model_path
            self.mm = model_class(output_dir=model_path)
            self.mm.load_model()

    def load_dict(self, user_profile_path=None,item_profile_path=None):
        if not user_profile_path:
            user_profile_path = self.user_profile_path
        with open(user_profile_path, 'r') as f:
            self.user_profile = json.load(f)

        if not item_profile_path:
            item_profile_path = self.item_profile_path
        with open(item_profile_path, 'r') as f:
            self.item_profile = json.load(f)

    def redis_connect(self, redis_info=None):
        if not redis_info:
            redis_info = self.redis_info
        pool = redis.ConnectionPool(**redis_info)
        self.redis_client = redis.Redis(connection_pool=pool)

    @staticmethod
    def batch_generator(dataset,batch_size,feature_names,key):
        """
        input:{'user_id':[user_tag1,user_tag2],...}
        output:
        an iter yileding
        {'user_id':[id1,id2,...],'user_tag1': tf.constant([value1,value2,...]), 'user_tag2': tf.constant([value1,value2])}
        with value length 'batch_size' at each iteration
        """
        _dataset = [[key, *value] for key, value in dataset.items()]
        _dataset=np.array(_dataset)
        data_size=_dataset.shape[0]
        """
        字典原来就无序，这里没有shuffle的必要
        if shuffle:
            # 随机生成打乱的索引
            p = np.random.permutation(data_size)
            # 重新组织数据
            _dataset = dataset[p,:]
        """
        batch_count=0
        end=0
        while end<data_size:
            batch_count+=1
            start=batch_count*batch_size
            end=start+batch_size
            result={field:tf.constant(_dataset[start:end,i].astype(int)) for i,field in enumerate(feature_names[1:],1)}
            result[key]=list(_dataset[start:end,0])
            yield result

    def generate_user_embedding(self,batch_size=10):
        """
        其实应该由线上来完成，不需要离线全量预测和存储
        实际：用户进入场景-->线上取实时特征和离线特征-->包括labelencode等的特征在线处理-->训练好的onnx文件-->线上出结果并使用
        这里：先存好用户特征(用户画像)-->特征处理并存储(只有labelencode)-->进模型出结果-->保存结果
        """
        self.user_embedding_dict = {}
        for input in self.batch_generator(self.user_profile,batch_size,self.user_feature_names,'user_id'):
            result=self.mm.layer.u_tower(input)
            user_ids=result['user_id']
            embeddings=result['output']
            for user_id,embedding in zip(user_ids,embeddings):
                self.user_embedding_dict[user_id] = [float(member.numpy()) for member in embedding]
        with open(self.ebd_save_path + 'user_embedding.json', 'w') as f:
            json.dump(self.user_embedding_dict,f)

    def generate_item_embedding(self,batch_size=10):
        """
        其实应该由线上来完成，不需要离线全量预测和存储
        实际：用户进入场景-->线上取实时特征和离线特征-->包括labelencode等的特征在线处理-->训练好的onnx文件-->线上出结果并使用
        这里：先存好用户特征(用户画像)-->特征处理并存储(只有labelencode)-->进模型出结果-->保存结果
        """
        self.item_embedding_dict={}
        for input in self.batch_generator(self.item_profile,batch_size,self.item_feature_names,'item_id'):
            result=self.mm.layer.i_tower(input)
            item_ids=result['item_id']
            embeddings=result['output']
            for item_id,embedding in zip(item_ids,embeddings):
                self.item_embedding_dict[item_id]=[float(member.numpy()) for member in embedding]
        with open(self.ebd_save_path+'item_embedding.json','w') as f:
            json.dump(self.item_embedding_dict,f)

    def build_ball_tree(self, cluster_center_num=None,ball_tree_path='retrieval_model/ball_tree/ball_tree_info.pkl'):
        if not cluster_center_num:
            cluster_center_num = self.cluster_center_num
        if len(self.item_embedding_dict) == 0:
            with open(self.ebd_save_path + '/item_embedding.json', 'r') as f:
                self.item_embedding_dict = json.load(f)
        if len(self.user_embedding_dict) == 0:
            with open(self.ebd_save_path + '/user_embedding.json', 'r') as f:
                self.user_embedding_dict = json.load(f)
        item_info = [(item_id, item_emb) for item_id, item_emb in self.item_embedding_dict.items()]
        # 存ball_tree的向量做归一化，以实现按内积计算取出的结果和按余弦相似度一样
        ball_tree = BallTree(np.array([info[1]/np.linalg.norm(info[1]) for info in item_info]), leaf_size=cluster_center_num)
        item_list = [info[0] for info in item_info]
        with open(ball_tree_path,'wb') as f:
            pickle.dump(ball_tree,f)
            pickle.dump(item_list,f)
        print('Ball tree built!')
        return ball_tree, item_list

    def load_to_redis(self, redis_info=None, cluster_center_num=None, fetch_num=None,ball_tree_path='retrieval_model/ball_tree/ball_tree_info.pkl'):
        redis_info = redis_info if redis_info else self.redis_info
        fetch_num = fetch_num if fetch_num else self.fetch_num
        cluster_center_num = cluster_center_num if cluster_center_num else self.cluster_center_num
        self.redis_connect(redis_info)
        ball_tree, item_list = self.build_ball_tree(cluster_center_num,ball_tree_path)
        redis_result = {}
        for user_id, user_emb in self.user_embedding_dict.items():
            dist, ind = ball_tree.query([user_emb], k=fetch_num)
            key = 'DSSM_' + user_id
            value = ','.join([item_list[id] for id in ind[0]]) + '\t' + ','.join(map(str, dist[0]))
            self.redis_client.set(key, value)
            redis_result[key] = value
        with open(self.ebd_save_path + '/u2i_retrieval_result.json', 'w') as f:
            json.dump(redis_result, f)

    def get_redis_value(self, key='DSSM_15394621775962514577', redis_info=None):
        redis_info = redis_info if redis_info else self.redis_info
        if not self.redis_client:
            self.redis_connect(redis_info)
        value = self.redis_client.get(key)
        with open(self.ebd_save_path + '/item_embedding.json', 'r') as f:
            self.item_embedding_dict = json.load(f)
        items=value.decode().split('\t')[0].split(',')
        print(value.decode().split('\t')[1])
        embeddings=[self.item_embedding_dict[item] for item in items]
        for i in embeddings:
            print(i)

    def run(self):
        self.load_model()
        self.load_dict()
        self.generate_item_embedding()
        self.generate_user_embedding()
        self.load_to_redis()
        self.get_redis_value()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse args.')
    parser.add_argument('--data_path', type=str, default='data/generated/mini_news-test-1')
    parser.add_argument('--model_path', type=str, default='retrieval_model/output')
    parser.add_argument('--o2i_dict_path', type=str, default='data/generated/o2i.json')
    parser.add_argument('--ebd_save_path', type=str, default='retrieval_model/emb_result/')
    parser.add_argument('--cluster_center_num', type=int, default=10)
    parser.add_argument('--fetch_num', type=int, default=20)
    parser.add_argument('--redis_info', type=str, default="{'host':'127.0.0.1', 'port':'6379', 'db':0}")
    args = parser.parse_args()
    offline_loader = OfflineLoader()
    offline_loader.run()
