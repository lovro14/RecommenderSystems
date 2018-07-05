import tensorflow as tf


class Model(object):

    def __init__(self, batch_size, latent_factor_dimension, learning_rate, user_number, item_number, iter_train,
                 dropout_rate=0.5, layers=None, min_rating=1, max_rating=5, reg_factor=0.1, transfer_learning=False,
                 optimizer_class=tf.train.AdamOptimizer, initializer=tf.truncated_normal_initializer(stddev=0.02),
                 random_state=None):
        self.batch_size = batch_size
        self.latent_factor_dimension = latent_factor_dimension
        self.learning_rate = learning_rate
        self.user_number = user_number
        self.item_number = item_number
        self.iter_train = iter_train
        self.dropout_rate = dropout_rate
        self.layers = layers
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.reg_factor = reg_factor
        self.transfer_learning = transfer_learning
        self.optimizer_class = optimizer_class
        self.initializer = initializer
        self._session = None
        self.random_state = random_state

    def _build_graph(self):
        pass

    def save(self, path):
        self.saver.save(self._session, path)

    def close_session(self):
        if self._session:
            self._session.close()

    def fit(self, user_ids_test, item_ids_test, ratings_test,
            rating_data_train, model_name, dataset, n_epoch=40, max_checkout_without_progress=20):
        pass
