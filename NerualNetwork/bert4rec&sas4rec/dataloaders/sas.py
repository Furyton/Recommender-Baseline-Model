from .base import AbstractDataloader

import torch
import torch.utils.data as data_utils
from copy import deepcopy


class SASDataLoader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.max_len = args.sas_max_len

    @classmethod
    def code(cls):
        return 'sas'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True, num_workers=self.worker_num)
        return dataloader

    def _get_train_dataset(self):
        dataset = SASTrainDataset(user_train=self.train, user_num=self.user_count, item_num=self.item_count,
                                  max_len=self.max_len, rng=self.rng)

        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True, num_workers=self.worker_num)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        # train_dataset = None
        if mode == 'val':
            train_dataset = deepcopy(self.train)
        else:
            train_dataset = deepcopy(self.train)
            for index, seq in enumerate(train_dataset):
                seq.append(self.val[index][0])

        dataset = SASEvalDataset(train_dataset, answers, self.max_len, self.test_negative_samples)
        return dataset


class SASTrainDataset(data_utils.Dataset):
    def __init__(self, user_train, user_num, item_num, max_len, rng):
        self.user_train = user_train
        self.user_num = user_num
        self.item_num = item_num
        self.max_len = max_len
        self.rng = rng
        self.users = range(len(user_train))

    def __len__(self):
        return len(self.user_train)

    def random_neq(self, l, r, s):
        t = self.rng.randint(l, r)
        while t in s:
            t = self.rng.randint(l, r)
        return t

    def __getitem__(self, index):
        user = self.users[index]

        # seq = np.zeros([self.max_len], dtype=np.int32)
        # os = np.zeros([self.max_len], dtype=np.int32)
        # neg = np.zeros([self.max_len], dtype=np.int32)

        seq = [0 for i in range(self.max_len)]
        pos = [0 for i in range(self.max_len)]
        neg = [0 for i in range(self.max_len)]
        nxt = self.user_train[user][-1]
        idx = self.max_len - 1

        ts = set(self.user_train[user])
        for i in reversed(self.user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = self.random_neq(1, self.item_num, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return torch.LongTensor(seq), torch.LongTensor(pos), torch.LongTensor(neg)


class SASEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, negative_samples):
        self.u2seq = u2seq
        self.users = range(len(u2seq))
        self.u2answer = u2answer
        self.max_len = max_len
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = deepcopy(self.u2seq[user])
        answer = deepcopy(self.u2answer[user])
        negs = deepcopy(self.negative_samples[user])

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)
