import numpy as np
from tqdm import tqdm
import itertools
import math


def get_similarity_list(observed_list: list, user_num, item_num):
    print("== calculating similarity matrix ==")

    sim_list = [{} for x in range(item_num)]

    occur = {}

    for items in tqdm(observed_list):
        for id in items:
            if id not in occur:
                occur[id] = 1
            else:
                occur[id] += 1

        for i, j in itertools.combinations(items, 2):
            if j not in sim_list[i]:
                sim_list[i][j] = 1
            else:
                sim_list[i][j] += 1

            if i not in sim_list[j]:
                sim_list[j][i] = 1
            else:
                sim_list[j][i] += 1

    for i in tqdm(range(item_num)):
        for key, value in sim_list[i].items():
            sim_list[i][key] /= math.sqrt(occur[i]) * math.sqrt(occur[key])

    return sim_list


# def get_similar_items(sim_matrix, item_num: int, k_neighbors: int):
#     print("== calculating similar items ==")
#
#     similar_items = []
#     for item_id in tqdm(range(item_num)):
#         similar_items.append(sorted(range(len(item_num)), key=lambda k: -sim_matrix[item_id][k])[1:k_neighbors + 1])
#
#     print("== done ==")
#
#     return similar_items

def get_k_similar_items(sim_list: list, item_num, k_neighbors):
    print("== calculating similar items ==")

    similar_items = []

    for item_id in tqdm(range(item_num)):
        # l =
        similar_items.append(
            [key for key, value in sorted(sim_list[item_id].items(), key=lambda item: -item[1])][:k_neighbors])

    print("== done ==")

    return similar_items


def test(observed_list: list, answer_list: list, candidates_list: list, occur: dict, user_num: int, item_num: int,
         k_neighbors: int):
    print("== testing ==")
    # minus 1, the item_id will start at 0
    observed_list = [[x - 1 for x in l] for l in observed_list]
    answer_list = [[x - 1 for x in l] for l in answer_list]
    candidates_list = [[x - 1 for x in l] for l in candidates_list]

    sim_list = get_similarity_list(observed_list=observed_list, user_num=user_num, item_num=item_num)

    similar_items = get_k_similar_items(sim_list=sim_list, item_num=item_num, k_neighbors=k_neighbors)

    # sim_matrix = similarity_matrix(observed_list, user_num=user_num, item_num=item_num)
    #
    # similar_items = get_similar_items(sim_matrix, item_num=item_num, k_neighbors=k_neighbors)

    rankings = []
    zeros = []

    # for observed, answer, candidates in tqdm(zip(observed_list, answer_list, candidates_list)):
    for i in tqdm(range(len(observed_list))):
        observed = observed_list[i]
        answer = answer_list[i]
        candidates = candidates_list[i]

        item_list = list(candidates)

        item_list.insert(np.random.randint(len(candidates)), answer[0])
        # item_list.insert(len(candidates), answer[0])

        scores = {}
        z = 0
        indices = []
        max_similar_sum = 0.0
        for item_id in item_list:
            similar_observed_ids = list(set(similar_items[item_id]).intersection(observed))
            # similar_sum = np.take(sim_matrix[item_id], similar_observed_ids)
            similar_sum = [sim_list[item_id][key] for key in similar_observed_ids]
            scores[item_id] = sum(similar_sum)

            if scores[item_id] > max_similar_sum:
                max_similar_sum = scores[item_id]

            if len(similar_observed_ids) == 0:
                z += 1

        zeros.append((scores[answer[0]], z))

        for k, v in scores.items():
            if v != 0.0:
                scores[k] = np.random.rand() * max_similar_sum

        scores = [key for key, value in sorted(scores.items(), key=lambda item: -item[1])]
        rankings.append(scores.index(answer[0]))

    print("== done ==")

    return rankings
