import tensorflow as tf

import numpy as np
import argparse

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 转载自网络，作为读取checkpint和从tensorboard观察loss变化的模板

num_epochs = 100
batch_size = 50
learning_rate = 0.001


class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)  # [60000]
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()  # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):  # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)  # [batch_size, 784]
        x = self.dense1(x)  # [batch_size, 100]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


data_loader = MNISTLoader()
log_dir = 'tensorboard'


def train():
    model = MLP()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    summary_writer = tf.summary.create_file_writer(log_dir)     # 实例化记录器
    #     tf.summary.trace_on(profiler=True)  # 开启Trace（可选）
    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
    # 使用tf.train.CheckpointManager管理Checkpoint
    manager = tf.train.CheckpointManager(checkpoint, directory='./save', max_to_keep=3)
    for batch_index in range(1, num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:

            # with tf.Session() as sess:
            #     sess.run(tf.global_variables_initializer())
            #     sess.run(tf.local_variables_initializer())
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            #                 loss = sess.run(loss)
            #                 y_pred, loss = sess.run([y_pred, loss])
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        #             print("batch %d: loss %f" % (batch_index, list(loss)))
            with summary_writer.as_default():                           # 指定记录器
                tf.summary.scalar("loss", loss, step=batch_index)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        if batch_index % 100 == 0:
            # 使用CheckpointManager保存模型参数到文件并自定义编号
            path = manager.save(checkpoint_number=batch_index)
            print("model saved to %s" % path)


#     with summary_writer.as_default():
#         tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)    # 保存Trace信息到文件（可选）


def test():
    model_to_be_restored = MLP()
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_restored)
    checkpoint.restore(tf.train.latest_checkpoint('./save'))
    y_pred = np.argmax(model_to_be_restored.predict(data_loader.test_data), axis=-1)
    print("test accuracy: %f" % (sum(y_pred == data_loader.test_label) / data_loader.num_test_data))


if __name__ == '__main__':
    #train()
    test()
#     if args.mode == 'train':
#         train()
#     if args.mode == 'test':
#         test()