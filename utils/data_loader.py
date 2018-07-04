import pandas as pd
import numpy as np
import os

class DataLoader(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def load_data(self):
        if self.dataset == 'ml-100k':
            self.rating_data = pd.read_csv(
                os.path.join(os.path.dirname(__file__), '../datasets/ml-100k/u.data'),
                sep='\t',
                engine="python",
                encoding="latin-1",
                names=['userid', 'itemid', 'rating', 'timestamp'])
        elif self.dataset == 'ml-1m':
            self.rating_data = pd.read_csv(
                os.path.join(os.path.dirname(__file__), '../datasets/ml-1m/ratings.dat'),
                sep='::',
                engine="python",
                encoding="latin-1",
                names=['userid', 'itemid', 'rating', 'timestamp'])
        else:
            self.rating_data = pd.read_csv(
                os.path.join(os.path.dirname(__file__), '../datasets/ml-20m/ratings.csv'),
                sep=',',
                header=None,
                names=['userId', 'movieId', 'rating', 'timestamp'],
                engine='python')
        self._prepare_dataset()

    def _prepare_dataset(self):
        if self.dataset == 'ml-100k' or self.dataset == 'ml-1m':
            self.rating_data["userid"] -= 1
            self.rating_data["itemid"] -= 1
            for column in self.rating_data:
                if column == "userid" or column == "itemid":
                    self.rating_data[column] = self.rating_data[column].astype(np.int32)
                if column == "rating":
                    self.rating_data[column] = self.rating_data[column].astype(np.float32)
        else:
            self.rating_data.drop(self.rating_data.index[[0]], inplace=True)
            for column in self.rating_data:
                if column == "userId" or column == "movieId":
                    self.rating_data[column] = self.rating_data[column].astype(np.int32)
                if column == "rating":
                    self.rating_data[column] = self.rating_data[column].astype(np.float32)
            self.rating_data["userId"] -= 1
            self.rating_data["movieId"] -= 1

    def train_test_split(self, train_data_size):
        rating_number = len(self.rating_data)
        rating_data = self.rating_data.iloc[np.random.permutation(rating_number)].reset_index(drop=True)
        index_split_rating_data = int(rating_number * train_data_size)
        rating_data_train = rating_data[0:index_split_rating_data]
        rating_data_test = rating_data[index_split_rating_data:].reset_index(drop=True)
        return rating_data_train, rating_data_test

    def get_test_data(self, input_data):
        inputs = np.transpose(np.vstack([np.array(input_data[i]) for i in range(len(input_data))]))
        user_item_rating_test = [inputs[:, i] for i in range(len(input_data))]
        return user_item_rating_test[0], user_item_rating_test[1], user_item_rating_test[2]

    def get_dataset_info(self):
        if self.dataset == 'ml-100k' or self.dataset == 'ml-1m':
            user_number = self.rating_data['userid'].max()
            item_number = self.rating_data['itemid'].max()
        else:
            user_number = self.rating_data['userId'].max()
            item_number = self.rating_data['movieId'].max()
        return user_number, item_number
