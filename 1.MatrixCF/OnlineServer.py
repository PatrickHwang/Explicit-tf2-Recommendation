from flask import Flask, request
import json
import Model
import redis
import tensorflow as tf

assert tf.__version__[0] == '2', 'model is built by tensorflow2,please check tf version first.'


class OnlineServer:
    def __init__(self, i2o_dict_path='data/generated/i2o.json',
                 redis_info={'host': '127.0.0.1', 'port': '6379', 'db': 0}, model_dir='model/output'):
        self.redis_info = redis_info
        with open(i2o_dict_path, 'r') as f:
            self.i2o_dict = json.load(f)
        self.redis_connect()
        self.model = Model.MatrixCF(output_dir=model_dir)
        self.model.load_model()

    def redis_connect(self, redis_info=None):
        if not redis_info:
            redis_info = self.redis_info
        pool = redis.ConnectionPool(**redis_info)
        self.redis_client = redis.Redis(connection_pool=pool)

    def retrieve(self, user_id='60414487256349587'):
        result = self.redis_client.get('MatrixCF_' + user_id)
        result = result.decode().split('\t')[0].split(',')
        return result

    def rank(self, user_id, retrieve_result):
        query = {'user': [[self.i2o_dict[user_id]]] * len(retrieve_result), \
                 'item': [[self.i2o_dict[item_id]] for item_id in retrieve_result]}
        prediction = self.model.infer(query)
        scores = prediction['pred'].numpy().tolist()
        scores = [score[0] for score in scores]
        result = list(zip(retrieve_result, scores))
        result.sort(key=lambda x: x[1], reverse=True)
        return result


app = Flask(__name__)
online_server = OnlineServer()


# curl http://127.0.0.1:5000/predict -d '{"user_id":"60414487256349587","type":"retrieve"}'
# curl http://127.0.0.1:5000/predict -d '{"user_id":"60414487256349587","type":"rank"}'
@app.route("/predict", methods=["POST"])
def infer():
    return_dict = {}
    if request.get_data() is None:
        return_dict["errcode"] = 1
        return_dict["errdesc"] = "data is None"

    data = json.loads(request.get_data())
    user_id = data.get("user_id", None)
    type = data.get("type", "retrieve")
    if user_id is not None:
        if type == "retrieve":
            res = online_server.retrieve(user_id)
            print(res)
            return json.dumps(res)
        if type == "rank":
            res = online_server.retrieve(user_id)
            res = online_server.rank(user_id, res)
            print(res)
            return json.dumps(res)


if __name__ == '__main__':
    # online_server = OnlineServer()
    # print(online_server.rank('60414487256349587',['9869859756723708280', '9350670896984183239', '10903028364823679100', '3945856908367334703', '13948016907605184205']))
    app.run(debug=True)
