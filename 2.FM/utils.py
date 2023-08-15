import pickle
import numpy as np

import tensorflow as tf

assert tf.__version__[0] == '2', 'model is built by tensorflow2,please check tf version first.'
import redis

from ModelManager import ModelManager


def check_tf_record(path='data/generated/e-Commerce-train-1', limit=3):
    ind = 1
    for serialized_example in tf.compat.v1.python_io.tf_record_iterator(path):
        ind += 1
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        sample = example.features.feature
        print(f"user_tag1: {sample['user_tag1']}")
        print(f"user_tag2: {sample['user_tag2']}")
        print(f"item_tag1: {sample['item_tag1']}")
        print(f"item_tag2: {sample['item_tag2']}")
        print(f"item_tag3: {sample['item_tag3']}")
        print(f"label: {sample['label']}")

        if ind > limit:
            break


def check_inference(model_manager):
    model_manager.load_checkpoint_and_eval(eval=False)
    test_ds = model_manager.init_dataset('test')
    test_ds = iter(test_ds)
    ds = next(test_ds)
    label = ds.pop("label")
    print('label', end=':')
    print(label)
    print('ds', end=':')
    print(ds)
    res = model_manager.infer(ds)
    print('res', end=':')
    print(res)


def get_redis_value(key='DSSM_9118615302969565273', redis_info={'host': '127.0.0.1', 'port': '6379', 'db': 0}):
    pool = redis.ConnectionPool(**redis_info)
    redis_client = redis.Redis(connection_pool=pool)
    value = redis_client.get(key)
    print(value.decode())


def batch_generator(dataset, batch_size, feature_names):
    """
    input:{'user_id':[user_tag1,user_tag2],...}
    output:
    an iter yielding
    {'user_id':[id1,id2,...],'user_tag1': tf.constant([value1,value2,...]), 'user_tag2': tf.constant([value1,value2])}
    with value length 'batch_size' at each iteration
    """
    _dataset = [[key, *value] for key, value in dataset.items()]
    _dataset = np.array(_dataset)
    data_size = _dataset.shape[0]
    batch_count = 0
    end = 0
    while end < data_size:
        batch_count += 1
        start = batch_count * batch_size
        end = start + batch_size
        result = {field: tf.constant(_dataset[start:end, i]) for i, field in enumerate(feature_names)}
        yield result


if __name__ == '__main__':
    # check_tf_record()
    # check_inference(ModelManager())
    # get_redis_value()
    dataset = {"7597230350533193880": [0, 16], "7097839246984260143": [0, 17], "4776308456495669413": [0, 18],
               "8974550734561865634": [0, 17], "13881651065552584948": [0, 17], "9999489160430663965": [0, 9],
               "3314176117917161714": [0, 17], "4340903754936026816": [0, 17], "1028614631604270633": [0, 19],
               "11946388591642466632": [0, 15], "16840828895427931786": [0, 4], "5049290349672742573": [0, 26],
               "5887276046594401981": [0, 5], "1300618480399971867": [0, 17], "16083010870528734162": [0, 17],
               "12944336195737269366": [0, 7], "15821832725244479903": [0, 18], "17156588072087677567": [0, 7],
               "11769919723486335641": [0, 5], "11842509869490899358": [0, 17], "1269364036214942871": [0, 9],
               "8284839996958703146": [0, 5], "13297944386758505664": [0, 9], "14177764420704582559": [0, 16],
               "1618605723556551028": [0, 17], "517855054289248362": [0, 18], "9294189359432650509": [0, 18],
               "10458527813875115179": [0, 10], "9108871545428923047": [0, 15], "8025390984198263136": [0, 5],
               "16774201124969663393": [0, 17], "9597485316116514945": [0, 17], "4414315610773024153": [0, 7],
               "16146953407751228215": [0, 12], "2934293305531055631": [0, 17], "9606708099242567447": [0, 25],
               "3841693014896430925": [0, 4], "16737698189342123689": [0, 17], "9681643102542595106": [0, 3],
               "14754869246603296821": [0, 16], "4153310838779082918": [0, 17], "17871975650352424419": [0, 4],
               "11003489158885784141": [0, 17], "8099194022227125527": [0, 5], "4911218326521799861": [0, 7],
               "16763377843509074089": [0, 10]}
    feature_names=['user_id','user_tag1','user_tag2']
    batch_size=8
    for result in batch_generator(dataset, batch_size, feature_names):
        print(result)