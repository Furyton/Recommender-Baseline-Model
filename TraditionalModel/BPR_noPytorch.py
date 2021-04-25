import numpy as np
from tqdm import tqdm
import pickle

latent_dim = 64
learning_rate = 0.001
epochs = 200
batch_size = 0
init_mean = 0
init_std = 0.1
reg_u = 0.0025
reg_i = 0.0025
reg_j = 0.00025
reg_bias = 0

p = None
q = None
bias = None


def create_factors(user_num, item_num):
    global p, q, bias
    p = np.random.normal(init_mean, init_std, (user_num, latent_dim))
    q = np.random.normal(init_mean, init_std, (item_num, latent_dim))
    bias = np.zeros(item_num, np.double)


def sampling(seen_items: list, item_num):
    return np.random.choice(seen_items), np.random.choice(list(set(range(item_num)) - set(seen_items)))


def predict_score(user, item):
    return np.dot(p[user], q[item])


def update_factors(u, i, j):
    x_uij = bias[i] - bias[j] + (predict_score(u, i) - predict_score(u, j))
    eps = 1 / (1 + np.exp(x_uij))

    bias[i] += learning_rate * (eps - reg_bias * bias[i])
    bias[j] += learning_rate * (-eps - reg_bias * bias[j])

    # Adjust the factors
    u_f = p[u]
    i_f = q[i]
    j_f = q[j]

    # Compute and apply factor updates
    p[u] += learning_rate * ((i_f - j_f) * eps - reg_u * u_f)
    q[i] += learning_rate * (u_f * eps - reg_i * i_f)
    q[j] += learning_rate * (-u_f * eps - reg_j * j_f)


def evaluate(answer_list: list, candidates_list: list, user_num: int):
    print("== evaluating ==")

    w = bias.T + np.dot(p, q.T)

    # pickle.dump([p, q, bias], open("parameters.p", "wb"))

    rankings = []

    for user_id in tqdm(range(user_num)):
        scores = {item_id: w[user_id][item_id] for item_id in answer_list[user_id] + candidates_list[user_id]}
        scores = [key for key, value in sorted(scores.items(), key=lambda item: -item[1])]
        rankings.append(scores.index(answer_list[user_id][0]))

    NDCG = {1: 0., 5: 0., 10: 0., 20: 0.}
    HIT = {1: 0., 5: 0., 10: 0., 20: 0.}
    MRR = {1: 0., 5: 0., 10: 0., 20: 0.}

    for rank in tqdm(rankings):
        for k, val in NDCG.items():
            if rank < k:
                NDCG[k] += 1. / np.log2(rank + 2)

        for k, val in HIT.items():
            if rank < k:
                HIT[k] += 1.

        for k, val in MRR.items():
            if rank < k:
                MRR[k] += 1. / (rank + 1)

    valid_user_num = len(rankings)

    MRR = {key: val / valid_user_num for key, val in MRR.items()}
    NDCG = {key: val / valid_user_num for key, val in NDCG.items()}
    HIT = {key: val / valid_user_num for key, val in HIT.items()}

    print('NDCG: ', NDCG)
    print('HIT: ', HIT)
    print('MRR: ', MRR)

    return NDCG, HIT, MRR


def test(observed_list: list, answer_list: list, candidates_list: list, user_num: int, item_num: int):
    print("== testing ==")
    # minus 1, the item_id will start at 0
    observed_list = [[x - 1 for x in l] for l in observed_list]
    answer_list = [[x - 1 for x in l] for l in answer_list]
    candidates_list = [[x - 1 for x in l] for l in candidates_list]

    create_factors(user_num=user_num, item_num=item_num)

    # pickle.dump([p, q, bias], open("parameters.p", "wb"))
    # fit(observed_items=observed_list, item_num=item_num)

    global p, q, bias
    pickle.dump([p, q, bias], open("parameters.p", "wb"))

    # global p, q, bias
    #
    # p, q, bias = pickle.load(open("parameters.p", "rb"))

    print("== fitting ==")

    last_hit = 0.

    for n in range(epochs):
        # for user_id, items in tqdm(enumerate(observed_items)):
        # print("\nepoch: ", n)

        for user_id in tqdm(range(len(observed_list)), desc='epoch: ' + str(n)):
            i, j = sampling(observed_list[user_id], item_num=item_num)
            update_factors(user_id, i, j)

        if n % 10 == 0:
            NDCG, HIT, MRR = evaluate(candidates_list=candidates_list, answer_list=answer_list, user_num=user_num)

            if HIT[10] > last_hit:
                last_hit = HIT[10]
                pickle.dump([p, q, bias], open("parameters.p", "wb"))

        # t.set_description("epoch: %" + str(n))

    print("== done ==")

    NDCG, HIT, MRR = evaluate(candidates_list=candidates_list, answer_list=answer_list, user_num=user_num)

    if HIT[10] > last_hit:
        last_hit = HIT[10]
        pickle.dump([p, q, bias], open("parameters.p", "wb"))
    #
    #
    # w = bias.T + np.dot(p, q.T)
    #
    # pickle.dump([p, q, bias], open("parameters.p", "wb"))
    #
    # rankings = []
    #
    # for user_id in tqdm(range(user_num)):
    #     scores = {item_id: w[user_id][item_id] for item_id in answer_list[user_id] + candidates_list[user_id]}
    #     scores = [key for key, value in sorted(scores.items(), key=lambda item: -item[1])]
    #     rankings.append(scores.index(answer_list[user_id][0]))
    #
    # print("== done ==")
    #
    # return rankings
