import pickle

import numpy as np
from .AbstractModel import BaseModel


class BPR_MF_model(BaseModel):
    def __init__(self, args, train: list, test: list, candidates: list, item_num, user_num):
        super().__init__()
        self.candidates = candidates
        self.d = args.dim
        self.iteration = args.iteration
        self.learning_rate = args.learning_rate
        self.regularization = args.regularization
        # self.std = args.std
        self.train = train
        self.test = test
        self.item_num = item_num
        self.user_num = user_num

        self.user_embedding = np.random.randn(self.user_num, self.d)
        self.item_embedding = np.random.randn(self.item_num, self.d)

        self.bias = np.zeros(self.item_num)

    def _update(self, user_index, positive_item, negative_item):
        user_factorization = np.copy(self.user_embedding[user_index, :])
        item_factorization_pos = np.copy(self.item_embedding[positive_item, :])
        item_factorization_neg = np.copy(self.item_embedding[negative_item, :])

        sigma = self._sigmoid(
            item_factorization_pos.T.dot(user_factorization) - item_factorization_neg.T.dot(user_factorization)
            + self.bias[positive_item] - self.bias[negative_item])

        c = 1.0 - sigma

        self.user_embedding[user_index, :] += self.learning_rate * (c * (item_factorization_pos
                                                                         - item_factorization_neg)
                                                                    - self.regularization * user_factorization)
        self.item_embedding[positive_item, :] += self.learning_rate * (c * (item_factorization_pos
                                                                            - item_factorization_neg)
                                                                       - self.regularization * item_factorization_pos)
        self.item_embedding[negative_item, :] += self.learning_rate * (-c * user_factorization
                                                                       - self.regularization * item_factorization_neg)
        self.bias[positive_item] += self.learning_rate * (sigma - self.regularization * self.bias[positive_item])
        self.bias[negative_item] += self.learning_rate * (-sigma - self.regularization * self.bias[negative_item])

        return np.log(sigma)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    @classmethod
    def code(cls):
        return 'bpr'

    def _sample(self, seen):
        return np.random.choice(seen), np.random.choice(list(set(range(self.item_num)) - set(seen)))

    def fit(self):
        for it in range(self.iteration):
            c = []
            for user_id in np.random.permutation(self.user_num):
                positive_item_id, negtive_item_id = self._sample(self.train[user_id])

                err = self._update(user_index=user_id, positive_item=positive_item_id, negative_item=negtive_item_id)

                c.append(err)

                # for i in np.random.permutation(len(self.train[user_id])):
                #     positive_item_id = self.train[user_id][i]
                #     negative_item_id = self._sample(self.train[user_id])
                #     err = self._update(user_index=user_id,
                #                        positive_item=positive_item_id, negative_item=negative_item_id)
                #     c.append(err)
            print(it, ' ', np.mean(c))

        pickle.dump([self.item_embedding, self.item_embedding], open("bpr_parameters.p", "wb"))

    def predict(self):
        rankings = []

        for user_id in range(self.user_num):
            scores = {item_id: self.user_embedding[user_id].T.dot(self.item_embedding[item_id])
                      for item_id in self.test[user_id] + self.candidates[user_id]}
            scores = [key for key, value in sorted(scores.items(), key=lambda item: -item[1])]

            rankings.append(scores.index(self.test[user_id][0]))

        return rankings
