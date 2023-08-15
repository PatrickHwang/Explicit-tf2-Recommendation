import json

import Model
import argparse
import redis
from sklearn.neighbors import BallTree
import tensorflow as tf
import pickle
import numpy as np

assert tf.__version__[0] == '2', 'model is built by tensorflow2,please check tf version first.'


class OfflineLoader:
    def __init__(self, data_path='data/generated/mini_news-test-1', model_path='model/output',
                 o2i_dict_path='data/generated/o2i.json',
                 ebd_save_path='model/emb_result/',
                 cluster_center_num=10, fetch_num=20, redis_info={'host': '127.0.0.1', 'port': '6379', 'db': 0}):
        self.model_path = model_path
        self.data_path = data_path
        self.o2i_dict_path = o2i_dict_path
        self.ebd_save_path = ebd_save_path
        self.cluster_center_num = cluster_center_num
        self.fetch_num = fetch_num
        self.redis_info = redis_info
        self.user_emb_dict = {}
        self.item_emb_dict = {}
        self.redis_client = None

    def load_model(self, model_path=None):
        if not model_path:
            model_path = self.model_path
        self.model = Model.MatrixCF(output_dir=model_path)
        self.model.load_model()

    def transform_data(self, data_path=None):
        if not data_path:
            data_path = self.data_path
        self.data = self.model.init_dataset('test', data_path, shuffle=False)

    def load_dict(self, o2i_dict_path=None):
        if not o2i_dict_path:
            o2i_dict_path = self.o2i_dict_path
        with open(o2i_dict_path, 'r') as f:
            self.o2i_dict = json.load(f)

    def redis_connect(self, redis_info=None):
        if not redis_info:
            redis_info = self.redis_info
        pool = redis.ConnectionPool(**redis_info)
        self.redis_client = redis.Redis(connection_pool=pool)

    def save_emb_results(self, model_path=None, data_path=None, o2i_dict_path=None,
                         ebd_save_path='model/emb_result/'):
        self.load_model(model_path)
        self.transform_data(data_path)
        self.load_dict(o2i_dict_path)
        data_iter = iter(self.data)
        while True:
            batch_data = next(data_iter, None)
            if not batch_data:
                print(f"The train dataset iterator is exhausted.")
                break
            batch_data.pop('ctr')
            user_codes = batch_data['user'].numpy()
            item_codes = batch_data['item'].numpy()
            output = self.model.infer(batch_data)
            user_embs = output['user_emb'].numpy()
            item_embs = output['item_emb'].numpy()
            for user_code, user_emb in zip(user_codes, user_embs):
                self.user_emb_dict[self.o2i_dict[str(user_code[0])]] = user_emb[0]
            for item_code, item_emb in zip(item_codes, item_embs):
                self.item_emb_dict[self.o2i_dict[str(item_code[0])]] = item_emb[0]
        with open(ebd_save_path + 'user_emb.pkl', 'wb') as f:
            pickle.dump(self.user_emb_dict, f)
        with open(ebd_save_path + 'item_emb.pkl', 'wb') as f:
            pickle.dump(self.item_emb_dict, f)

    def build_ball_tree(self, cluster_center_num=None):
        if not cluster_center_num:
            cluster_center_num = self.cluster_center_num
        if len(self.item_emb_dict) == 0:
            with open(self.ebd_save_path + 'user_emb.pkl', 'rb') as f:
                self.user_emb_dict = pickle.load(f)
            with open(self.ebd_save_path + 'item_emb.pkl', 'rb') as f:
                self.item_emb_dict = pickle.load(f)
        item_info = [(item_id, item_emb) for item_id, item_emb in self.item_emb_dict.items()]
        ball_tree = BallTree(np.array([info[1] for info in item_info]), leaf_size=cluster_center_num)
        item_list = [info[0] for info in item_info]
        return ball_tree, item_list

    def load_to_redis(self, redis_info=None, cluster_center_num=None, fetch_num=None):
        redis_info = redis_info if redis_info else self.redis_info
        fetch_num = fetch_num if fetch_num else self.fetch_num
        cluster_center_num = cluster_center_num if cluster_center_num else self.cluster_center_num
        self.redis_connect(redis_info)
        ball_tree, item_list = self.build_ball_tree(cluster_center_num)
        redis_result = {}
        for user_id, user_emb in self.user_emb_dict.items():
            dist, ind = ball_tree.query([user_emb], k=fetch_num)
            key = 'MatrixCF_' + user_id
            value = ','.join([self.o2i_dict[id] for id in map(str, ind[0])]) + '\t' + ','.join(map(str, dist[0]))
            self.redis_client.set(key, value)
            redis_result[key] = value
        with open(self.ebd_save_path + 'retrieval_info.json', 'w') as f:
            json.dump(redis_result, f)

    def get_redis_value(self, key='MatrixCF_60414487256349587', redis_info=None):
        redis_info = redis_info if redis_info else self.redis_info
        if not self.redis_client:
            self.redis_connect(redis_info)
        value = self.redis_client.get(key)
        print(value)

    def run(self):
        self.save_emb_results()
        self.load_to_redis()
        print('checking value of key MatrixCF_60414487256349587')
        self.get_redis_value()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse args.')
    parser.add_argument('--data_path', type=str, default='data/generated/mini_news-test-1')
    parser.add_argument('--model_path', type=str, default='model/output')
    parser.add_argument('--o2i_dict_path', type=str, default='data/generated/o2i.json')
    parser.add_argument('--ebd_save_path', type=str, default='model/emb_result/')
    parser.add_argument('--cluster_center_num', type=int, default=10)
    parser.add_argument('--fetch_num', type=int, default=20)
    parser.add_argument('--redis_info', type=str, default="{'host':'127.0.0.1', 'port':'6379', 'db':0}")
    args = parser.parse_args()
    offline_loader = OfflineLoader(data_path=args.data_path, model_path=args.model_path,
                                   o2i_dict_path=args.o2i_dict_path,
                                   cluster_center_num=args.cluster_center_num,
                                   fetch_num=args.fetch_num, redis_info=eval(args.redis_info))
    offline_loader.run()
