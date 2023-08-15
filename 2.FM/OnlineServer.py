import pickle

from flask import Flask, request
import json
from ModelManager import ModelManager
import redis
import tensorflow as tf

assert tf.__version__[0] == '2', 'model is built by tensorflow2,please check tf version first.'


class OnlineServer:
    """DSSM双塔召回和DeepFm排序"""
    def __init__(self, info_path='data/generated',
                 redis_info={'host': '127.0.0.1', 'port': '6379', 'db': 0},
                 feature_names=['user_tag1', 'user_tag2','item_tag1','item_tag2','item_tag3'],
                 user_feature_num=2,retrieval_layer='dssm_double_tower',ranking_layer='deepfm_ranking'):
        self.redis_info = redis_info
        self.info_path = info_path
        self.feature_names=feature_names
        self.user_feature_num=user_feature_num

        self.retrieval_mm = ModelManager(layer=retrieval_layer)
        self.retrieval_mm.load_checkpoint_and_eval(eval=False)
        self.ranking_mm = ModelManager(layer=ranking_layer,checkpoint_dir='ranking_model/checkpoint',
                                       tensorboard_dir='ranking_model/tensorboard',output_dir='ranking_model/output')
        self.ranking_mm.load_checkpoint_and_eval(eval=False)
        self.redis_connect()
        self.load_profile()
        self.ball_tree=None
        self.item_list=None

    def redis_connect(self, redis_info=None):
        if not redis_info:
            redis_info = self.redis_info
        pool = redis.ConnectionPool(**redis_info)
        self.redis_client = redis.Redis(connection_pool=pool)

    def load_profile(self):
        item_path=self.info_path+'/item_profile.json'
        user_path=self.info_path+'/user_profile.json'
        with open(item_path,'r') as f:
            self.item_profile=json.load(f)
        with open(user_path,'r') as f:
            self.user_profile=json.load(f)

    def retrieve(self, user_id='15394621775962514577'):
        result = self.redis_client.get('DSSM_' + user_id)
        result = result.decode().split('\t')[0].split(',')
        print(result)
        return result

    def retrieve_online(self,user_id='4877416706748717568',fetch_num=20):
        # 双塔召回，用户embedding在线进模型生成，商品embedding离线存于ball_tree或faiss
        # 0.加载离线生成好的ball_tree和item_list
        if not self.ball_tree:
            with open('retrieval_model/ball_tree/ball_tree_info.pkl', 'rb') as f:
                self.ball_tree=pickle.load(f)
                self.item_list=pickle.load(f)
        # 1.线上取用户实时特征+离线特征，进用户塔模型，输出用户embedding
        user_feature_names = self.feature_names[:self.user_feature_num]
        user_feature_values = self.user_profile[str(user_id)]
        X = {feature: [] for feature in user_feature_names}
        for feature_name, feature_value in zip(user_feature_names, user_feature_values):
            X[feature_name].append(feature_value)
        X = {key: tf.constant(value) for key, value in X.items()}
        user_embedding = self.retrieval_mm.layer.u_tower(X)['output'].numpy()[0].tolist()
        # 2.从ball_tree中查找最相似的结果
        dist, ind = self.ball_tree.query([user_embedding], k=fetch_num)
        items = [self.item_list[id] for id in ind[0]]
        dist = dist[0]
        # 3.根据得分截断、过滤等各种业务逻辑处理，最终给到排序器
        print(items)
        print(dist)
        return items

    def rank(self, user_id='15394621775962514577', retrieval_result=['16249142194827819179','10838150046406483301','14235907696851131833'],from_retreival=False):
        if from_retreival:
            retrieval_result=self.retrieve_online(user_id)
            print(retrieval_result)
        X={feature:[] for feature in self.feature_names}
        user_feature_names=self.feature_names[:self.user_feature_num]
        item_feature_names=self.feature_names[self.user_feature_num:]
        user_feature_values=self.user_profile[str(user_id)]
        for feature_name,feature_value in zip(user_feature_names,user_feature_values):
            X.update({feature_name:[feature_value]*len(retrieval_result)})
        for item_id in retrieval_result:
            item_feature_values=self.item_profile[str(item_id)]
            for feature_name, feature_value in zip(item_feature_names, item_feature_values):
                X[feature_name].append(feature_value)
        X={key:tf.constant(value) for key,value in X.items()}

        # model和layer不一样,会包装input的形状,不能互相替换使用
        res = self.ranking_mm.model(X)
        res = res['output'].numpy().tolist()
        print(res)
        result = {}
        for i,item_id in enumerate(retrieval_result):
            result[item_id] = res[i][0]
        print(result)
        return result



app = Flask(__name__)
online_server = OnlineServer()

# curl http://127.0.0.1:5000/predict -d '{"user_id":"15394621775962514577","type":"retrieve","source":"redis"}'
# curl http://127.0.0.1:5000/predict -d '{"user_id":"15394621775962514577","type":"retrieve"}'
# curl http://127.0.0.1:5000/predict -d '{"user_id":"15394621775962514577","item_ids":['16249142194827819179','10838150046406483301','14235907696851131833'],"type":"rank"}'
# curl http://127.0.0.1:5000/predict -d '{"user_id":"15394621775962514577","type":"rank","from_retrieval":"true"}'
@app.route("/predict", methods=["POST"])
def infer():
    return_dict = {}
    if request.get_data() is None:
        return_dict["errcode"] = 1
        return_dict["errdesc"] = "data is None"

    data = json.loads(request.get_data())
    user_id = data.get("user_id", None)
    type = data.get("type", "retrieve")
    item_ids=data.get("item_ids",None)

    retrieve_source=data.get('source','ball_tree')
    false=False
    true=True
    from_retrieval=eval(data.get('from_retrieval',"False"))

    if user_id is not None:
        if type == "retrieve":
            if retrieve_source=='redis':
                res = online_server.retrieve(user_id)
            else:
                res = online_server.retrieve_online(user_id)
            print(res)
            return json.dumps(res)
        elif type == "rank":
            res = online_server.rank(user_id,retrieval_result=item_ids,from_retreival=from_retrieval)
            print(res)
            return json.dumps(res)


if __name__ == '__main__':
    app.run(debug=True)
    # online_server.retrieve()
    # online_server.rank()
    # online_server.rank(from_retreival=True)