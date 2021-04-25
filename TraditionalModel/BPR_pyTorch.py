import numpy as np
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

# epochs = 500
lamb = 0.025  # 正则化系数


# lr = 1e-3
# dim = 32


class BPRData(data.Dataset):
    def __init__(self, dataset, item_num, user_num, negative_num, data_len, is_training=None):
        super(BPRData, self).__init__()

        self.processed_data = []
        self.raw_data = dataset
        self.item_num = item_num
        self.user_num = user_num
        self.is_training = is_training
        self.negative_num = negative_num
        self.data_len = data_len

    def negative_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        for user_id, observed_items in enumerate(self.raw_data):
            for i in observed_items:
                for t in range(self.negative_num):
                    j = np.random.randint(self.item_num)
                    while j in observed_items:
                        j = np.random.randint(self.item_num)
                    self.processed_data.append([user_id, i, j])

    def __len__(self):
        return self.data_len * self.negative_num if self.is_training else self.data_len

    def __getitem__(self, idx):
        # features = self.processed_data if self.is_training else self.raw_data
        #
        if self.is_training:
            return self.processed_data[idx]
        else:
            return idx, self.raw_data[idx]


def get_train_dataset(train_data, item_num, user_num, data_len, negative_num=1):
    return BPRData(train_data, item_num, user_num, negative_num, data_len, True)


def get_train_loader(train_dataset, batch_size=4096):
    return data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(BPR, self).__init__()

        self.embed_user = nn.Embedding(user_num + 1, factor_num, sparse=True)
        self.embed_item = nn.Embedding(item_num + 1, factor_num, sparse=True)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)

        return prediction_i, prediction_j, lamb * (
                (user * user).sum(dim=-1) + (item_i * item_i).sum(dim=-1) + (item_j * item_j).sum(dim=-1))


def evaluate(ranks: list):
    # print("== evaluating ==")

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

    # print("== done ==")

    return NDCG, HIT, MRR


def train(train_data, item_num, user_num, data_len, answer_list, candidates_list, epoch_num, lr, dim, device_num,
          candidate_num,
          model_dir=None, device='cpu'):
    print("== training ==")
    # train_data = [[x - 1 for x in l] for l in train_data]
    # answer_list = [[x - 1 for x in l] for l in answer_list]
    # candidates_list = [[x - 1 for x in l] for l in candidates_list]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_num)
    train_loader = get_train_loader(get_train_dataset(train_data, item_num, user_num, data_len, negative_num=100))

    model = BPR(user_num, item_num, dim)

    if model_dir is not None:
        model = torch.load(model_dir, map_location=torch.device(device))

    model = model.to(device=device)

    model.eval()

    rankings = []

    last_HIT = 0

    for user_id in tqdm(range(user_num)):
        items = torch.tensor(answer_list[user_id] + candidates_list[user_id]).to(dtype=torch.int64, device=device)
        user = torch.tensor([user_id] * (candidate_num + 1)).to(dtype=torch.int64, device=device)

        with torch.no_grad():
            scores = model(user, items, items)
            scores = {items[i].item(): scores[0][i].item() for i in range((candidate_num + 1))}
            # scores = {item_id: w[user_id][item_id] for item_id in answer_list[user_id] + candidates_list[user_id]}
            scores = [key for key, value in sorted(scores.items(), key=lambda item: -item[1])]
            rankings.append(scores.index(answer_list[user_id][0]))

    NDCG, HIT, MRR = evaluate(rankings)

    print('initial metrics:')
    print('NDCG: ', NDCG)
    print('HIT: ', HIT)
    print('MRR: ', MRR)

    last_HIT = HIT[10]

    optimizer = optim.SparseAdam(model.parameters(), lr=lr)

    for epoch in range(epoch_num):
        model.train()
        train_loader.dataset.negative_sample()
        for user, item_i, item_j in tqdm(train_loader):
            user = user.to(device=device)
            item_i = item_i.to(device=device)
            item_j = item_j.to(device=device)
            model.zero_grad()
            prediction_i, prediction_j, reg = model(user, item_i, item_j)
            loss = -(prediction_i - prediction_j).sigmoid().log().sum() + reg.sum()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            model.eval()

            rankings = []

            for user_id in tqdm(range(user_num)):
                items = torch.tensor(answer_list[user_id] + candidates_list[user_id]).to(dtype=torch.int64,
                                                                                         device=device)
                user = torch.tensor([user_id] * (candidate_num + 1)).to(dtype=torch.int64, device=device)

                with torch.no_grad():
                    scores = model(user, items, items)
                    scores = {items[i].item(): scores[0][i].item() for i in range((candidate_num + 1))}
                    # scores = {item_id: w[user_id][item_id] for item_id in answer_list[user_id] + candidates_list[user_id]}
                    scores = [key for key, value in sorted(scores.items(), key=lambda item: -item[1])]
                    rankings.append(scores.index(answer_list[user_id][0]))

            NDCG, HIT, MRR = evaluate(rankings)

            print('NDCG: ', NDCG)
            print('HIT: ', HIT)
            print('MRR: ', MRR)

            if HIT[10] > last_HIT:
                last_HIT = HIT[10]
                torch.save(model, 'best_HIT10.pth')

    torch.save(model, 'model.pth')

    model.eval()

    for user_id in tqdm(range(user_num)):
        items = torch.tensor(answer_list[user_id] + candidates_list[user_id]).to(dtype=torch.int64, device=device)
        user = torch.tensor([user_id] * (candidate_num + 1)).to(dtype=torch.int64, device=device)

        with torch.no_grad():
            scores = model(user, items, items)
            scores = {items[i].item(): scores[0][i].item() for i in range((candidate_num + 1))}
            # scores = {item_id: w[user_id][item_id] for item_id in answer_list[user_id] + candidates_list[user_id]}
            scores = [key for key, value in sorted(scores.items(), key=lambda item: -item[1])]
            rankings.append(scores.index(answer_list[user_id][0]))

    return evaluate(rankings)
