import tensorflow as tf

assert tf.__version__[0] == '2', 'model is built by tensorflow2,please check tf version first.'
import Model
import redis


def check_tf_record(path='/Volumes/Beta/deepshare/CustomCoding/1.MatrixCF/data/generated/mini_news-test-1', limit=1):
    ind = 1
    for serialized_example in tf.compat.v1.python_io.tf_record_iterator(path):
        ind += 1
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        sample = example.features.feature
        print(f"user: {sample['user']}")
        print(f"item: {sample['item']}")
        print(f"ctr: {sample['ctr']}")

        if ind > limit:
            break


def check_inference():
    mcf = Model.MatrixCF(batch=3)
    mcf.load_model()
    print(mcf.imported)
    test_ds = mcf.init_dataset('test')
    test_ds = iter(test_ds)
    ds = next(test_ds)
    label = ds.pop("ctr")
    print('label', end=':')
    print(label)
    print('ds', end=':')
    print(ds)
    res = mcf.infer(ds)
    print('res', end=':')
    print(res)


def get_redis_value(key='MatrixCF_60414487256349587', redis_info={'host': '127.0.0.1', 'port': '6379', 'db': 0}):
    pool = redis.ConnectionPool(**redis_info)
    redis_client = redis.Redis(connection_pool=pool)
    value = redis_client.get(key)
    print(value.decode())


if __name__ == '__main__':
    # check_tf_record()
    # check_inference()
    get_redis_value()
