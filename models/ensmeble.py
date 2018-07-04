from models.model import Model
import tensorflow as tf
import numpy as np
from utils.utilities import Utility
from collections import deque
import time
from tensorflow.core.framework import summary_pb2
import os


class EnsembleModel(Model):
    def __init__(self, *args, **kwargs):
        super(EnsembleModel, self).__init__(*args, **kwargs)

    def _create_dnn(self, inputs, layers, dropout_rate=None, training=None, batch_norm_momentum=None):
        for i in range(len(layers)):
            if dropout_rate:
                inputs = tf.layers.dropout(inputs, dropout_rate, training=training)

            name = "hidden%d" % (i + 1)
            if self.transfer_learning:
                inputs = tf.layers.dense(inputs, layers[i],
                                         kernel_initializer=tf.constant_initializer(self.neural_network_kernels[i]),
                                         name=name)
            else:
                inputs = tf.layers.dense(inputs, layers[i], kernel_initializer=self.initializer, name=name)

            if batch_norm_momentum:
                inputs = tf.layers.batch_normalization(inputs, momentum=batch_norm_momentum, training=training)

            inputs = tf.nn.relu(inputs)
        return inputs

    def _build_graph(self):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        user = tf.placeholder(tf.int32, shape=[None], name="user_id")
        item = tf.placeholder(tf.int32, shape=[None], name="item_id")
        rating = tf.placeholder(tf.float32, shape=[None], name="rating")

        if self.transfer_learning:
            w_user_mf = tf.get_variable(
                "embd_user_mf",
                initializer=tf.constant_initializer(self.w_user_mf_transfer_learning),
                shape=self.w_user_mf_transfer_learning.shape)
            w_item_mf = tf.get_variable(
                "embd_item_mf",
                initializer=tf.constant_initializer(self.w_item_mf_transfer_learning),
                shape=self.w_item_mf_transfer_learning.shape)
            w_user_mlp = tf.get_variable(
                "embd_user_mlp",
                initializer=tf.constant_initializer(self.w_user_mlp_transfer_learning),
                shape=self.w_user_mlp_transfer_learning.shape)
            w_item_mlp = tf.get_variable(
                "embd_item_mlp",
                initializer=tf.constant_initializer(self.w_item_mlp_transfer_learning),
                shape=self.w_item_mlp_transfer_learning.shape)
        else:
            w_user_mf = tf.get_variable("embd_user_mf", shape=[self.user_number + 1, self.latent_factor_dimension],
                                        initializer=self.initializer)
            w_item_mf = tf.get_variable("embd_item_mf", shape=[self.item_number + 1, self.latent_factor_dimension],
                                        initializer=self.initializer)
            w_user_mlp = tf.get_variable("embd_user_mlp", shape=[self.user_number + 1, self.latent_factor_dimension],
                                         initializer=self.initializer)
            w_item_mlp = tf.get_variable("embd_item_mlp", shape=[self.item_number + 1, self.latent_factor_dimension],
                                         initializer=self.initializer)

        embd_user_mf = tf.nn.embedding_lookup(w_user_mf, user, name="embeddings_user_mf")
        embd_item_mf = tf.nn.embedding_lookup(w_item_mf, item, name="embeddings_item_mf")
        mf_prediction = tf.multiply(embd_user_mf, embd_item_mf)

        embd_user_mlp = tf.nn.embedding_lookup(w_user_mlp, user, name="embeddings_user_mlp")
        embd_item_mlp = tf.nn.embedding_lookup(w_item_mlp, item, name="embeddings_item_mlp")
        training = tf.placeholder_with_default(False, shape=(), name="training")
        mlp_input = tf.concat([embd_user_mlp, embd_item_mlp], axis=1)
        mlp_prediction = self._create_dnn(mlp_input, self.layers, self.dropout_rate, training)

        neomf_input = tf.concat([mf_prediction, mlp_prediction], axis=1)
        prediction = tf.layers.dense(neomf_input, 1, kernel_initializer=self.initializer, name="neomf")
        prediction = tf.reshape(prediction, [-1])

        loss = tf.nn.l2_loss(tf.subtract(prediction, rating))

        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self.prediction = prediction
        self.training_op = training_op
        self.loss = loss
        self.saver = saver
        self.init = init
        self.user = user
        self.item = item
        self.rating = rating

    def fit(self, user_ids_test, item_ids_test, ratings_test, rating_data_train, model_name, dataset, n_epoch=40):
        self.batch_number = len(rating_data_train) // self.batch_size

        self.close_session()
        if self.transfer_learning:
            latenFactorModelPath = os.path.join(os.path.dirname(__file__),
                                                '../pretrained_models/' + dataset,
                                                'latent-factor-model-' + dataset + '.ckpt')
            restore_saver = tf.train.import_meta_graph(latenFactorModelPath+".meta")
            with tf.Session() as sess:
                restore_saver.restore(sess, latenFactorModelPath)
                self.w_user_mf_transfer_learning = tf.get_default_graph().get_tensor_by_name("embd_user:0").eval()
                self.w_item_mf_transfer_learning = tf.get_default_graph().get_tensor_by_name("embd_item:0").eval()

            Utility.reset_graph()

            deepNeuralNetworkModelPath = os.path.join(os.path.dirname(__file__),
                                                      '../pretrained_models/' + dataset,
                                                      'deep-neural-network-model-'+dataset+'.ckpt')
            restore_saver_dnn = tf.train.import_meta_graph(deepNeuralNetworkModelPath + ".meta")
            with tf.Session() as sess:
                restore_saver_dnn.restore(sess, deepNeuralNetworkModelPath)
                self.neural_network_kernels = []
                self.w_user_mlp_transfer_learning = tf.get_default_graph().get_tensor_by_name("embd_user_mlp:0").eval()
                self.w_item_mlp_transfer_learning = tf.get_default_graph().get_tensor_by_name("embd_item_mlp:0").eval()
                for i in range(len(self.layers)):
                    self.neural_network_kernels.append(
                        tf.get_default_graph().get_tensor_by_name("hidden"+str(i+1)+"/kernel:0").eval())


            Utility.reset_graph()

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph()


        max_checkout_without_progress = 100
        checkout_without_progress = 0
        min_test_error = np.infty
        self._session = tf.Session(graph=self._graph)

        with self._session.as_default() as sess:
            self.init.run()
            path = os.path.join(os.path.dirname(__file__), '../tf_logs/' + dataset)
            logdir = Utility.log_dir(path=path, prefix=model_name)
            summary_writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)
            print("Epoch\tloss\ttrain_error\tRMSE\tMAE\tAccuracy\ttime")
            errors = deque(maxlen=self.batch_number)

            for epoch in range(n_epoch):
                start = time.time()
                errors = []
                losses = []
                for iteration in range(self.batch_number):
                    user_ids, item_ids, ratings = next(self.iter_train)
                    _, loss_val, prediction_batch = sess.run(
                        [self.training_op, self.loss, self.prediction],
                        feed_dict={self.user: user_ids, self.item: item_ids, self.rating: ratings})
                    prediction_batch = np.clip(prediction_batch, self.min_rating, self.max_rating)
                    errors.append(np.power(prediction_batch - ratings, 2))
                    losses.append(loss_val)

                train_err = np.sqrt(np.mean(errors))
                loss_mean = np.mean(losses)
                prediction_test_data = sess.run([self.prediction], feed_dict={self.user: user_ids_test, self.item: item_ids_test})
                prediction_test_data = np.clip(prediction_test_data, self.min_rating, self.max_rating)

                accuracy = np.sum(np.round(np.asarray(prediction_test_data)) == ratings_test) / len(ratings_test)
                mae = np.mean(np.abs(prediction_test_data - ratings_test))

                test_err = np.sqrt(np.mean(np.power(prediction_test_data - ratings_test, 2)))
                end = time.time()
                if (test_err < min_test_error):
                    min_test_error = test_err
                    saver_path = self.saver.save(sess, os.path.join(os.path.dirname(__file__),
                                                                    '../pretrained_models/' + dataset + '/' + model_name))
                    checkout_without_progress = 0
                    final_prediction_test_data = sess.run([self.prediction],
                                                          feed_dict={self.user: user_ids_test, self.item: item_ids_test})
                    self.final_prediction_test_data = np.clip(final_prediction_test_data, self.min_rating, self.max_rating)
                else:
                    checkout_without_progress += 1
                    if checkout_without_progress >= max_checkout_without_progress:
                        print("Early stopping")
                        break
                duration = end - start
                print("{:3d}\t{}\t{}\t{}\t{}\t{}\t{}(s)".format(epoch, loss_mean, train_err, test_err, mae, accuracy,
                                                                duration))

                loss_summary = summary_pb2.Summary(
                    value=[summary_pb2.Summary.Value(tag="loss", simple_value=loss_mean)])
                summary_writer.add_summary(loss_summary, epoch)

                train_error_summary = summary_pb2.Summary(
                    value=[summary_pb2.Summary.Value(tag="training_error", simple_value=train_err)])
                summary_writer.add_summary(train_error_summary, epoch)

                test_error_summary = summary_pb2.Summary(
                    value=[summary_pb2.Summary.Value(tag="RMSE", simple_value=test_err)])
                summary_writer.add_summary(test_error_summary, epoch)

                accuracy_summary = summary_pb2.Summary(
                    value=[summary_pb2.Summary.Value(tag="accuracy", simple_value=accuracy)])
                summary_writer.add_summary(accuracy_summary, epoch)

                mae_summary = summary_pb2.Summary(
                    value=[summary_pb2.Summary.Value(tag="MAE", simple_value=mae)])
                summary_writer.add_summary(mae_summary, epoch)

    def get_test_data_prediction(self):
        return self.final_prediction_test_data
