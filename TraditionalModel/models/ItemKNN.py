import numpy as np
from .AbstractModel import BaseModel


class itemKNN_model(BaseModel):
    def __init__(self, args, train: list, test: list, candidates: list, item_num, user_num):
        super().__init__()
        self.k = args.k
        self.lmbd = args.lmbd
        self.alpha = args.alpha
        self.train = train
        self.test = test
        self.candidates = candidates
        self.item_num = item_num
        self.user_num = user_num
        self.item_count = np.zeros(item_num)
        for items in train:
            for item_id in items:
                self.item_count[item_id] += 1
        self.sim = {}

    @classmethod
    def code(cls):
        return 'knn'

    def fit(self):
        self.similarity_matrix = dict()
        for items in self.train:
            for item_i in items:
                for item_j in items:
                    if item_i not in self.similarity_matrix:
                        self.similarity_matrix[item_i] = {}
                    elif item_j not in self.similarity_matrix[item_i]:
                        self.similarity_matrix[item_i][item_j] = 1
                    else:
                        self.similarity_matrix[item_i][item_j] += 1
        for item_id in range(self.item_num):
            self.similarity_matrix[item_id][item_id] = 0

            iarray = np.zeros(self.item_num)
            iarray[np.array(list(self.similarity_matrix[item_id].keys()))] = list(self.similarity_matrix[item_id].values())
            norm = np.power((self.item_count[item_id] + self.lmbd), self.alpha) * np.power((np.array(self.item_count) + self.lmbd), (1.0 - self.alpha))
            # norm = np.power((supp[i] + self.lmbd), self.alpha) * np.power((supp.values + self.lmbd), (1.0 - self.alpha))
            norm[norm == 0] = 1
            iarray = iarray / norm
            indices = np.argsort(iarray)[-1:-1-self.k:-1]
            self.sim[item_id] = np.array([indices, iarray[indices]])

    def predict(self):
        rankings = []
        for user_id, items in enumerate(self.train):
            candidate_items = self.candidates[user_id] + self.test[user_id]
            preds = np.zeros(len(candidate_items))

            index = np.random.randint(len(items))

            for item_id in items[index:index+2]:
                sim = self.sim[item_id]
                preds[np.in1d(candidate_items, sim[0])] += sim[1][np.in1d(sim[0], candidate_items)]

            scores = {candidate_items[i]: preds[i] for i in range(len(candidate_items))}

            # scores = {item_id: self.user_embedding[user_id].T.dot(self.item_embedding[item_id])
            #                       for item_id in self.test[user_id] + self.candidates[user_id]}
            scores = [key for key, value in sorted(scores.items(), key=lambda item: -item[1])]
            #
            rankings.append(scores.index(self.test[user_id][0]))
        return rankings