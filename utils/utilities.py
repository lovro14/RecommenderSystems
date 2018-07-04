from datetime import datetime
import tensorflow as tf
import numpy as np


class Utility(object):

    def __init__(self):
        pass

    @staticmethod
    def log_dir(path, prefix=""):
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_logdir = path
        if prefix:
            prefix += "-"
        name = prefix + "run-" + now
        return "{}/{}/".format(root_logdir, name)

    @staticmethod
    def reset_graph(seed=42):
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)