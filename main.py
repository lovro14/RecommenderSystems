from utils.config import parse_args
from utils.data_loader import DataLoader
import tensorflow as tf
from models.latent_factor_model import LatentFactorModel
from models.deep_neural_network_model import DeepNeuralNetworkModel
from models.ensmeble import EnsembleModel
from utils.shuffle_iterator import ShuffleIterator
from evaluator.recommender_evaluator import RecommenderEvaluator


def main(args):
    dataset = args.dataset
    model_type = args.model
    layers = []
    if model_type != 'latent-factor-model':
        layers = eval(args.layers)
    n_epoch = args.epochs
    max_checkout_without_progress = args.max_checkout_without_progress
    batch_size = args.batch_size
    dimension = args.dimension
    learning_rate = args.learning_rate
    if args.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer
    elif args.optimizer == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer
    else:
        optimizer = tf.train.GradientDescentOptimizer
    dropout_rate = args.dropout_rate
    regularization_factor = args.regularization_factor

    data_loader = DataLoader(dataset)
    data_loader.load_data()
    user_number, item_number = data_loader.get_dataset_info()
    rating_data_train, rating_data_test = data_loader.train_test_split(0.8)

    iter_train = ShuffleIterator([rating_data_train["userid"], rating_data_train["itemid"],
                                  rating_data_train["rating"]], batch_size=batch_size)

    if dataset == 'ml-100k' or dataset == 'ml-1m':
        userid = "userid"
        itemid = "itemid"
    else:
        userid = "userId"
        itemid = "movieId"

    user_ids_test, item_ids_test, ratings_test = data_loader.get_test_data([rating_data_test[userid],
                                                rating_data_test[itemid], rating_data_test["rating"]])
    model_name = model_type + '-' + dataset
    if model_type == 'latent-factor-model':
        model = LatentFactorModel(batch_size, dimension, learning_rate, user_number, item_number, iter_train,
                              dropout_rate, optimizer_class=optimizer, reg_factor=regularization_factor)
    elif model_type == 'deep-neural-network-model':
        model = DeepNeuralNetworkModel(batch_size, dimension, learning_rate, user_number, item_number, iter_train,
                              dropout_rate, layers=layers, optimizer_class=optimizer, reg_factor=regularization_factor)
    elif model_type == 'ensemble-no-transfer-learning':
        model = EnsembleModel(batch_size, dimension, learning_rate, user_number, item_number, iter_train,
                              dropout_rate, layers=layers[:-1], optimizer_class=optimizer, reg_factor=regularization_factor)
    else:
        model = EnsembleModel(batch_size, dimension, learning_rate, user_number, item_number, iter_train,
                              dropout_rate, layers=layers[:-1], optimizer_class=optimizer, reg_factor=regularization_factor,
                              transfer_learning=True)

    model.fit(user_ids_test, item_ids_test, ratings_test,
              rating_data_train, model_name, dataset, n_epoch=n_epoch, max_checkout_without_progress=max_checkout_without_progress)

    predicted_ratings = model.get_test_data_prediction()
    evaluator = RecommenderEvaluator(rating_data_test, predicted_ratings, dataset)
    print("\nRMSE={}".format(evaluator.rmse()))
    print("MAE={}".format(evaluator.mae()))
    k = 20
    precisions, recalls = evaluator.precision_recall_at_k(k)
    precision = sum(prec for prec in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)
    f1 = evaluator.f1(precision, recall)
    print("Precision({})={}".format(k, precision))
    print("Recall({})={}".format(k, recall))
    print("F1({})={}".format(k, f1))


if __name__ == '__main__':
    args = parse_args()
    main(args)
