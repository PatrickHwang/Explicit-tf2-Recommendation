import argparse
import os
import tensorflow as tf
from abc import ABC,abstractmethod

assert tf.__version__[0] == '2', 'model is built by tensorflow2,please check tf version first.'


class EarlyStopper:
    """
    跟踪模型训练中每个epoch的指标，触发早停条件时返回信号停止训练
    """
    def __init__(self, num_trials, is_higher_better=True):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.is_higher_better = is_higher_better
        if is_higher_better:
            self.best_metric = 0
        else:
            self.best_metric = 1e6

    def is_continuable(self, metric):
        if (self.is_higher_better and metric > self.best_metric) or \
                (not self.is_higher_better and metric < self.best_metric):
            self.best_metric = metric
            self.trial_counter = 0
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


class ModelBase(ABC):
    def __init__(self, feature_dims=9000, embedding_size=16, lr=0.01, label_name='ctr',
                 checkpoint_dir='model/checkpoint',
                 data_dir='data/generated', output_dir='model/output', n_threads=-1, shuffle=50, batch=16, epochs=30,
                 restore=False, num_trials=3):
        self.feature_dims = feature_dims
        self.embedding_size = embedding_size
        self.lr = lr
        self.label_name = label_name
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_threads = n_threads
        self.shuffle = shuffle
        self.batch = batch
        self.epochs = epochs
        self.restore = restore
        self.num_trials = num_trials
        # 初始化类时一同初始化建模所需对象
        self.init_model()
        self.init_loss()
        self.init_opt()
        self.init_metric()
        self.init_save_checkpoint()
        # self.train_dataset=self.init_dataset('train')
        # self.test_dataset=self.init_dataset('test')

    # 定义model
    @abstractmethod
    def init_model(self):
        pass

    # 定义loss
    def init_loss(self):
        # mse或mae也可以
        self.loss = tf.keras.losses.BinaryCrossentropy()

    # 定义optam
    def init_opt(self):
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)

    # 定义metric
    def init_metric(self):
        self.metric_auc = tf.keras.metrics.AUC(name="auc")
        self.metric_loss = tf.keras.metrics.Mean(name="loss")

    # 保存模型
    def init_save_checkpoint(self):
        checkpoint = tf.train.Checkpoint(optimizer=self.opt, model=self.model)
        self.manager = tf.train.CheckpointManager(
            checkpoint,
            directory=self.checkpoint_dir,
            max_to_keep=3,
            checkpoint_interval=True,
            step_counter=self.opt.iterations)

    # 准备输入数据
    @abstractmethod
    def init_dataset(self, type, file_dir=None, shuffle=True):
        pass

    # train
    def train_step(self, ds):
        # start
        def train_loop_begin():
            self.metric_auc.reset_states()
            self.metric_loss.reset_states()

        # end
        def train_loop_end():
            result = {
                self.metric_auc.name: self.metric_auc.result().numpy(),
                self.metric_loss.name: self.metric_loss.result().numpy()
            }
            return result

        # loop
        def train_loop(inputs):
            with tf.GradientTape() as tape:
                target = inputs.pop(self.label_name)
                logits = self.model(inputs, training=True)
                scaled_loss = tf.reduce_sum(self.loss(target, logits["pred"]))
                gradients = tape.gradient(scaled_loss,
                                          self.model.trainable_variables)
                self.opt.apply_gradients(
                    list(zip(gradients, self.model.trainable_variables)))
                self.metric_loss.update_state(scaled_loss)
                self.metric_auc.update_state(target, logits["pred"])

        # training
        step = 0
        inputs = iter(ds)
        train_loop_begin()
        while True:
            batch_data = next(inputs, None)
            if not batch_data:
                print(f"The train dataset iterator is exhausted after {step} steps.")
                break
            train_loop(batch_data)  # yield的是一个batch的数据
            step += 1
            if step % 500 == 0:
                result = train_loop_end()
                print(f"train over-{step} : {result}")
        result = train_loop_end()
        print(f"train over-{step} : {result}")
        return result

    # eval
    def eval_step(self, ds):
        # start
        def eval_loop_begin():
            self.metric_auc.reset_states()

        # end
        def eval_loop_end():
            result = {self.metric_auc.name: self.metric_auc.result().numpy()}
            return result

        # loop
        def eval_loop(inputs):
            target = inputs.pop(self.label_name)
            logits = self.model(inputs, training=True)
            self.metric_auc.update_state(target, logits["pred"])

        # eval
        step = 0
        inputs = iter(ds)
        eval_loop_begin()
        while True:
            batch_data = next(inputs, None)
            if not batch_data:
                print(f"The eval dataset iterator is exhausted after {step} steps.")
                break
            eval_loop(batch_data)
            step += 1
            if step % 500 == 0:
                result = eval_loop_end()
                print(f"eval over-{step} : {result}")
        result = eval_loop_end()
        print(f"eval over-{step} : {result}")
        return result

    # run
    def run(self, train_ds, test_ds, mode="train_and_eval", incremental=False):
        if incremental:
            self.load_model()
            self.model = self.imported
        if self.restore:
            self.manager.restore_or_initialize()
        early_stopper = EarlyStopper(num_trials=self.num_trials)
        if mode == "train_and_eval" and train_ds is not None and test_ds is not None:
            for epoch in range(self.epochs):
                print(f"=====epoch: {epoch}=====")
                train_result = self.train_step(train_ds)
                eval_result = self.eval_step(test_ds)
                if not early_stopper.is_continuable(eval_result[self.metric_auc.name]):
                    print(
                        f'validation best accuracy {early_stopper.best_metric} at epoch {epoch - early_stopper.num_trials}')
                    break
                if eval_result[self.metric_auc.name] == early_stopper.best_metric:
                    self.manager.save(checkpoint_number=epoch,
                                      check_interval=True)
                    
        if mode == "train":
            for epoch in range(self.epochs):
                print(f"=====epoch: {epoch}=====")
                train_result = self.train_step(train_ds)
                if train_result[self.metric_auc.name] > 0.5:
                    self.manager.save(checkpoint_number=epoch,
                                      check_interval=True)
        if mode == "eval":
            eval_result = self.eval_step(test_ds)

    # export
    def export_model(self):
        tf.saved_model.save(self.model, self.output_dir)

    def load_model(self):
        self.imported = tf.saved_model.load(self.output_dir)
        # self.f = imported.signatures["serving_default"]

    # infer
    def infer(self, x):
        result = self.imported(x)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse args.')
    parser.add_argument('--feature_dims', type=int, default=9000)
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--label_name', type=str, default='ctr')
    parser.add_argument('--checkpoint_dir', type=str, default='model/checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/generated')
    parser.add_argument('--output_dir', type=str, default='model/output')
    parser.add_argument('--n_threads', type=int, default=-1)
    parser.add_argument('--shuffle', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--restore', type=bool, default=False)
    parser.add_argument('--num_trials', type=int, default=3)
    args = parser.parse_args()
    mcf = ModelBase(feature_dims=args.feature_dims, embedding_size=args.embedding_size, lr=args.lr,
                   label_name=args.label_name, checkpoint_dir=args.checkpoint_dir,
                   data_dir=args.data_dir, output_dir=args.output_dir, n_threads=args.n_threads, shuffle=args.shuffle,
                   batch=args.batch, epochs=args.epochs, restore=args.restore, num_trials=args.num_trials)
    train_ds = mcf.init_dataset('train')
    test_ds = mcf.init_dataset('test', shuffle=False)
    # train
    mcf.run(train_ds, test_ds, mode="train_and_eval", incremental=False)
    # save
    mcf.export_model()
