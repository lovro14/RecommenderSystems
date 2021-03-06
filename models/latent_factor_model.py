from models.model import Model
import tensorflow as tf
import numpy as np
from utils.utilities import Utility
from collections import deque
import time
from tensorflow.core.framework import summary_pb2
import os


class LatentFactorModel(Model):

    def __init__(self, *args, **kwargs):
        super(LatentFactorModel, self).__init__(*args, **kwargs)

    def _build_graph(self):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        user = tf.placeholder(tf.int32, shape=[None], name="user_id")
        item = tf.placeholder(tf.int32, shape=[None], name="item_id")
        rating = tf.placeholder(tf.float32, shape=[None], name="rating")

        bias_global = tf.get_variable("bias_global", shape=[])
        w_bias_user = tf.get_variable("emdb_bias_user", shape=[self.user_number+1])
        w_bias_item = tf.get_variable("embd_bias_item", shape=[self.item_number+1])
        bias_user = tf.nn.embedding_lookup(w_bias_user, user, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item, item, name="bias_item")
        w_user = tf.get_variable("embd_user", shape=[self.user_number+1, self.latent_factor_dimension],
                                 initializer=self.initializer)
        w_item = tf.get_variable("embd_item", shape=[self.item_number+1, self.latent_factor_dimension],
                                 initializer=self.initializer)
        embd_user = tf.nn.embedding_lookup(w_user, user, name="embeddings_user")
        embd_item = tf.nn.embedding_lookup(w_item, item, name="embeddings_item")

        mf_prediction = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)

        mf_prediction = tf.add(mf_prediction, bias_global)
        mf_prediction = tf.add(mf_prediction, bias_user)
        mf_prediction = tf.add(mf_prediction, bias_item, name="MF")
        reguralizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="regularizer")

        global_step = tf.train.get_global_step()

        loss_l2 = tf.nn.l2_loss(tf.subtract(mf_prediction, rating))
        lambda_penalty = tf.constant(self.reg_factor, dtype=tf.float32, shape=[], name="l2")
        loss = tf.add(loss_l2, tf.multiply(reguralizer, lambda_penalty))
        #loss = loss_l2
        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss, global_step=global_step)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self.mf_prediction = mf_prediction
        self.training_op = training_op
        self.saver = saver
        self.init = init
        self.user = user
        self.item = item
        self.rating = rating

    def fit(self, user_ids_test, item_ids_test, ratings_test, rating_data_train, model_name, dataset, n_epoch=40,
            max_checkout_without_progress=20):
        self.batch_number = len(rating_data_train) // self.batch_size

        self.close_session()

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph()

        checkout_without_progress = 0
        min_test_error = np.infty
        latent_factor_rmse = []
        latent_factor_mae = []
        self._session = tf.Session(graph=self._graph)

        with self._session.as_default() as sess:
            self.init.run()
            path = os.path.join(os.path.dirname(__file__), '../tf_logs/'+dataset)
            logdir = Utility.log_dir(path=path, prefix=model_name)
            summary_writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)
            print("Epoch\ttrain_error\tRMSE\tMAE\tAccuracy\ttime")
            errors = deque(maxlen=self.batch_number)

            for epoch in range(n_epoch):
                start = time.time()
                for iteration in range(self.batch_number):
                    user_ids, item_ids, ratings = next(self.iter_train)
                    _, prediction_batch = sess.run(
                        [self.training_op, self.mf_prediction],
                        feed_dict={self.user: user_ids, self.item: item_ids, self.rating: ratings})
                    prediction_batch = np.clip(prediction_batch, self.min_rating, self.max_rating)
                    errors.append(np.power(prediction_batch - ratings, 2))

                train_err = np.sqrt(np.mean(errors))
                prediction_test_data = sess.run([self.mf_prediction], feed_dict={self.user: user_ids_test, self.item: item_ids_test})
                prediction_test_data = np.clip(prediction_test_data, self.min_rating, self.max_rating)

                mf_accuracy = np.sum(np.round(np.asarray(prediction_test_data)) == ratings_test) / len(ratings_test)
                mae = np.mean(np.abs(prediction_test_data - ratings_test))

                test_err = np.sqrt(np.mean(np.power(prediction_test_data - ratings_test, 2)))
                end = time.time()
                if (test_err < min_test_error):
                    min_test_error = test_err
                    saver_path = self.saver.save(sess, os.path.join(os.path.dirname(__file__),
                                                                    '../pretrained_models/'+dataset+'/'+model_name+".ckpt"))
                    #print("Model saved in file: %s" % saver_path)
                    checkout_without_progress = 0
                    final_prediction_test_data = sess.run([self.mf_prediction],
                                                          feed_dict={self.user: user_ids_test, self.item: item_ids_test})
                    self.final_prediction_test_data = np.clip(final_prediction_test_data, self.min_rating, self.max_rating)
                else:
                    checkout_without_progress += 1
                    if checkout_without_progress >= max_checkout_without_progress:
                        print("Early stopping")
                        break
                duration = end - start
                print("{:3d}\t{}\t{}\t{}\t{}\t{}(s)".format(epoch, train_err, test_err, mae, mf_accuracy, duration))

                train_error_summary = summary_pb2.Summary(
                    value=[summary_pb2.Summary.Value(tag="training_error", simple_value=train_err)])
                summary_writer.add_summary(train_error_summary, epoch)

                test_error_summary = summary_pb2.Summary(
                    value=[summary_pb2.Summary.Value(tag="RMSE", simple_value=test_err)])
                summary_writer.add_summary(test_error_summary, epoch)
                latent_factor_rmse.append(test_err)
                latent_factor_mae.append(mae)
                mf_accuracy_summary = summary_pb2.Summary(
                    value=[summary_pb2.Summary.Value(tag="accuracy", simple_value=mf_accuracy)])
                summary_writer.add_summary(mf_accuracy_summary, epoch)

                mae_summary = summary_pb2.Summary(
                    value=[summary_pb2.Summary.Value(tag="MAE", simple_value=mae)])
                summary_writer.add_summary(mae_summary, epoch)

    def get_test_data_prediction(self):
        return self.final_prediction_test_data

