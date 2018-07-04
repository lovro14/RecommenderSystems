import numpy as np
from collections import defaultdict


class RecommenderEvaluator(object):
    def __init__(self, rating_data_test, predicted_ratings, dataset):
        self._prepare_results_data(dataset, rating_data_test, predicted_ratings)

    def _prepare_results_data(self, dataset, rating_data_test, predicted_ratings):
        if dataset == "ml-100k" or dataset == "ml-1m":
            self.results = rating_data_test[["userid", "itemid", "rating", "timestamp"]].copy()
        else:
            self.results = rating_data_test[["userId", "movieId", "rating", "timestamp"]].copy()
        self.results["predicted_rating_round"] = np.round(predicted_ratings[0])
        self.results["predicted_rating_decimal"] = predicted_ratings[0]

    def get_results_data(self):
        return self.results

    def rmse(self):
        return np.sqrt(np.mean(np.power(self.results["rating"] - self.results["predicted_rating_decimal"], 2)))

    def mae(self):
        return np.mean(np.abs(self.results["rating"] - self.results["predicted_rating_decimal"]))

    def precision_recall_at_k(self, k=20):
        user_est_true = defaultdict(list)
        for index, row in self.results.iterrows():
            user_est_true[row['userid']].append((row['predicted_rating_decimal'], row['rating']))

        mean_test = {}
        for uid, user_ratings in user_est_true.items():
            count = 0
            sum_ratings = 0
            for pred_real in user_ratings:
                count = count + 1
                sum_ratings = sum_ratings + pred_real[1]
            mean_test[uid] = sum_ratings / count

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():
            if len(user_ratings) >= k:
                user_ratings.sort(key=lambda x: x[0], reverse=True)
                n_rel = sum((true_r >= mean_test[uid]) for (_, true_r) in user_ratings)
                n_rec_k = sum((est >= mean_test[uid]) for (est, _) in user_ratings[:k])
                n_rel_and_rec_k = sum(((true_r >= mean_test[uid]) and (est >= mean_test[uid]))
                                      for (est, true_r) in user_ratings[:k])
                precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
                recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

        return precisions, recalls

    def f1(self, precision, recall):
        return 2 * precision * recall / (precision + recall)


