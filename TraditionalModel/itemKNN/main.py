import ast
import BPR_pyTorch
import numpy as np
from tqdm import tqdm
import math
import pickle

PATH = 'data/ml-1m/pop_test.txt'


# :return a dictionary recording the user and the corresponding interacted items

def get_data():
    x = {}
    with open(PATH, 'r') as f:
        contents = f.readlines()
        contents = [[int(x.split('\t')[0]), ast.literal_eval(x.split('\t')[1])] for x in contents]
        # x = x.split('\t')
        # print(ast.literal_eval(x[1]))
    x = {x: y for x, y in contents}
    return x
    # return y


def get_item_num(data_dict):
    max_item_id = 0
    for user_id, items in data_dict.items():
        for item_id in items:
            if item_id > max_item_id:
                max_item_id = item_id

    return max_item_id


# return 2 lists

def split_dataset(data_dict):
    observed = []
    predict = []

    for user_id, items in data_dict.items():
        observed.append(items[:-1])
        predict.append([items[-1]])

    return observed, predict


# data is a list
# return 1 dict

def get_popularity(data, max_item_id):
    popularity = {x: 0 for x in range(1, max_item_id + 1)}

    for items in data:
        for item_id in items:
            popularity[item_id] += 1

    return popularity


# sorted_items is a list
# return a list

def select_candidates(data_dict, sorted_items_dict: dict, candidate_num=100):
    print("== selecting candidates ==")
    candidates = []

    keys = np.array([x for x in sorted_items_dict.keys()])
    values = sorted_items_dict.values()
    sum_value = np.sum([x for x in values])
    probability = [value * 1.0 / sum_value for value in values]

    for user_id, items in tqdm(data_dict.items()):
        cur_candidates = []

        while len(cur_candidates) < candidate_num:
            candidate_ids = np.random.choice(keys, candidate_num, replace=False, p=probability)
            candidate_ids = [x for x in candidate_ids if x not in items and x not in cur_candidates]
            cur_candidates.extend(candidate_ids[:])
        candidates.append(cur_candidates[:candidate_num])

    print("== done ==")

    return candidates


# def select_candidates(data_dict, sorted_items_dict: dict, candidate_num=100):
#     print("== selecting candidates ==")
#     candidates = []
#
#     top = list(sorted_items_dict.keys())
#
#     for user_id, items in tqdm(data_dict.items()):
#         cur_candidates = []
#         pointer = 0
#         for i in range(candidate_num):
#             while top[pointer] in items:
#                 pointer += 1
#             cur_candidates.append(top[pointer])
#             pointer += 1
#         candidates.append(cur_candidates)
#
#     print("== done ==")
#
#     return candidates


def evaluate(ranks: list):
    print("== evaluating ==")

    NDCG = {1: 0., 5: 0., 10: 0., 20: 0.}
    HIT = {1: 0., 5: 0., 10: 0., 20: 0.}
    MRR = {1: 0., 5: 0., 10: 0., 20: 0.}

    for rank in tqdm(ranks):
        for k, val in NDCG.items():
            if rank < k:
                NDCG[k] += 1. / np.log2(rank + 2)

        for k, val in HIT.items():
            if rank < k:
                HIT[k] += 1.

        for k, val in MRR.items():
            if rank < k:
                MRR[k] += 1. / (rank + 1)

    valid_user_num = len(ranks)

    MRR = {key: val / valid_user_num for key, val in MRR.items()}
    NDCG = {key: val / valid_user_num for key, val in NDCG.items()}
    HIT = {key: val / valid_user_num for key, val in HIT.items()}

    print("== done ==")

    return NDCG, HIT, MRR


def get_data_len(dataset):
    tot_len = 0
    for items in dataset:
        tot_len += len(items)
    return tot_len


if __name__ == '__main__':
    data = get_data()

    user_num = len(data)

    max_item_id = get_item_num(data)

    print("== splitting dataset ==")

    observed, predict = split_dataset(data)

    popularity = get_popularity(observed, max_item_id)

    sorted_items_dict = dict(sorted(popularity.items(), key=lambda item: -item[1]))

    #candidates = select_candidates(data_dict=data, sorted_items_dict=sorted_items_dict, candidate_num=100)

    # pickle.dump(candidates, open('data_ml.p', 'wb'))

    candidates = pickle.load(open('data_ml.p', 'rb'))

    BPR_pyTorch.train(observed, max_item_id, user_num, get_data_len(observed), predict, candidates)

    # BPRFM.test(observed_list=observed, answer_list=predict, candidates_list=candidates,
    #                           user_num=user_num, item_num=max_item_id)
    # predict_rank = itemKNN.test(observed_list=observed, answer_list=predict, candidates_list=candidates,
    #                             user_num=user_num, item_num=max_item_id, k_neighbors=int(math.sqrt(max_item_id)), occur=popularity)

    # predict_rank = POP.test(popularity=popularity, observed_list=observed, answer_list=predict, candidates_list=candidates)

    # NDCG, HIT, MRR = evaluate(predict_rank)
    #
    # print('NDCG: ', NDCG)
    # print('HIT: ', HIT)
    # print('MRR: ', MRR)
