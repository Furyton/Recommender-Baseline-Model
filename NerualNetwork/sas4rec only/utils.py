import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict, Counter
from multiprocessing import Process, Queue


# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(0, usernum)
        while len(user_train[user]) <= 1: user = np.random.randint(0, usernum)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return user, seq, pos, neg

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = []
    user_valid = []
    user_test = []
    # assume user/item index starting from 1
    f = open('data/%s' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    data = []

    for user in User:
        if len(User[user]) < 3:
            continue
        # if len(User[user]) <= max_len:
        data.append(User[user])

    usernum = len(data)

    for i in range(usernum):
        if len(data[i]) < 3:
            print("HERE")
        user_train.append(data[i][:-2])
        # user_valid[i] = []
        user_valid.append([data[i][-2]])
        # user_test[i] = []
        user_test.append([data[i][-1]])

    return [user_train, user_valid, user_test, usernum, itemnum]


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    MRR1 = 0.0
    MRR5 = 0.0
    MRR10 = 0.0
    MRR20 = 0.0
    NDCG20 = 0.0
    HT20 = 0.0
    NDCG10 = 0.0
    HT10 = 0.0
    NDCG5 = 0.0
    HT5 = 0.0
    HT1 = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(0, usernum), 10000)
    else:
        users = range(usernum)

    collection = Counter()
    for u in users:
        collection.update(train[u])
        collection.update(valid[u])

    keys = np.array([x for x in collection.keys()])
    values = collection.values()
    sum_value = np.sum([x for x in values])
    probability = [value / sum_value for value in values]

    # users = range(usernum)
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        while len(item_idx) < 101:
            sampled_ids = np.random.choice(keys, 101, replace=False, p=probability)
            sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
            item_idx.extend(sampled_ids[:])
        item_idx = item_idx[:101]
            # t = np.random.randint(1, itemnum + 1)
            # while t in rated: t = np.random.randint(1, itemnum + 1)
            # item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 20:
            NDCG20 += 1 / np.log2(rank + 2)
            HT20 += 1
            MRR20 += 1 / (rank + 1)
        if rank < 10:
            NDCG10 += 1 / np.log2(rank + 2)
            HT10 += 1
            MRR10 += 1 / (rank + 1)
        if rank < 5:
            NDCG5 += 1 / np.log2(rank + 2)
            HT5 += 1
            MRR5 += 1 / (rank + 1)
        if rank < 1:
            HT1 += 1
            MRR1 += 1 / (rank + 1)
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG20 / valid_user, HT20 / valid_user, MRR20 / valid_user, \
           NDCG10 / valid_user, HT10 / valid_user,  MRR10 / valid_user, \
           NDCG5 / valid_user, HT5 / valid_user, MRR5 / valid_user, \
           HT1 / valid_user, MRR1 / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(usernum), 10000)
    else:
        users = range(usernum)
    # users = range(usernum)

    collection = Counter()
    for u in users:
        collection.update(train[u])
        collection.update(valid[u])

    keys = np.array([x for x in collection.keys()])
    values = collection.values()
    sum_value = np.sum([x for x in values])
    probability = [value / sum_value for value in values]

    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        while len(item_idx) < 101:
            sampled_ids = np.random.choice(keys, 101, replace=False, p=probability)
            sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
            item_idx.extend(sampled_ids[:])
        item_idx = item_idx[:101]
        # t = np.random.randint(1, itemnum + 1)
        # while t in rated: t = np.random.randint(1, itemnum + 1)
        # item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user