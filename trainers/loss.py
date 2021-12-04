import numpy as np
from scipy.fftpack import diff
import torch.nn as nn
import torch
import torch.nn.functional as F

def neg_sample(seq, labels, num_item, sample_size):
    negs = set()
    seen = set(labels)

    while len(negs) < sample_size:
        candidate = np.random.randint(0, num_item) + 1
        while candidate in seen or candidate in negs:
            candidate = np.random.randint(0, num_item) + 1
        negs.add(candidate)
    return negs
    # keys = range(1, num_item + 1)
    # sample_id = np.random.choice(keys, sample_size + len(seen), replace=False)
    # sample_ids = [x for x in sample_id if x not in seen]

    # return sample_ids[:sample_size]


class LE:
    def __init__(self, model, args):
        self.model = model
        self.b = args.alpha

        self.t = args.T
        self.enable_sample = args.enable_sample
        self.num_item = args.num_items
        self.sample_ratio = args.samples_ratio
        if self.enable_sample:
            self.ce = nn.CrossEntropyLoss()
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.device = args.device
        self.debug = 0

        # self.b = torch.tensor(0., device=self.device, requires_grad=True)

    def compute(self, pred, batch):

        # seqs, labels, rating = batch
        seqs = batch[0]
        labels = batch[1]

        with torch.no_grad():
            soft_target = self.model(batch).detach().clone()  # B * L * N or B * N

        # we want both pred and soft_target to be 2-D tensor, like B * N, cl is a tensor which the size at 1st dim is the same, like B * 1

        if len(pred.size()) == 3: 
            # B * L * N, mask type
            cl = labels[labels > 0]
            pred = pred[labels > 0]

            assert(len(soft_target.size()) == 3)

            soft_target = soft_target[labels > 0]
        else:
            # B * N, next type
            cl = labels.squeeze()
            if len(soft_target.size()) == 3:
                # B * L * N
                soft_target = soft_target[:, -1, :].squeeze()
            else:
                # B * N
                soft_target
        
        assert(pred.size() == soft_target.size())

        _KL_Loss = 0.
        _CE_Loss = 0.

        _A_KL_Loss = 0.
        _A_CE_Loss = 0.

        _al = 0.

        if self.enable_sample:
            raise NotImplementedError("[enable_sample] not fully tested yet")
            # 负样本采样 
            negs = neg_sample(seqs, labels, self.num_item, int(self.num_item * self.sample_ratio))
            negs = torch.LongTensor(list(negs)).repeat(len(cl), 1).to(self.device)
            target = torch.cat((cl.unsqueeze(1), negs), 1)
            # 采样后的one_hot
            one_hot = [1] + [0] * negs.size(-1)
            one_hot = torch.LongTensor(one_hot).repeat(negs.size(0), 1).to(torch.device(self.device))
            # 抽取采样后的结果
            pred = pred.gather(dim=1, index=target)
            soft_target = soft_target.gather(dim=1, index=target)

            # print(f"[loss self.enable_sample] soft_target.size(): {soft_target.size()}")

            # 标签
            label = torch.LongTensor([0] * pred.size(0)).to(torch.device(self.device))
            soft_target = ((soft_target - soft_target.mean(dim=-1).unsqueeze(-1)) / soft_target.std(dim=-1).unsqueeze(-1))
            # 计算kl的值
            # KL_Loss = nn.functional.kl_div((pred.softmax(dim=-1) / self.t).log(), 0.5 * ((soft_target.softmax(dim=-1) / self.t) + one_hot), reduction='batchmean')
            KL_Loss = nn.functional.kl_div(F.log_softmax(pred, dim=-1), 0.5 * ((soft_target / self.t).softmax(dim=-1) + one_hot),  reduction='batchmean')

            if ~torch.isinf(KL_Loss):
                tmp_ce = self.ce(pred, label)
                # alpha = 0.2 * torch.sigmoid(self.b)
                alpha = self.b

                # _al = alpha.item()
                _al = alpha

                # tmp_kl = pow(self.t, 2) * alpha * KL_Loss
                tmp_kl = alpha * KL_Loss

                loss = (1 - alpha) * tmp_ce

                _KL_Loss = KL_Loss.item()
                _CE_Loss = tmp_ce.item()
                _A_CE_Loss = loss.item()
                _A_KL_Loss = tmp_kl.item()

                loss += tmp_kl
            else:
                loss = self.ce(pred, label)
                _CE_Loss = _A_CE_Loss = loss.item()
        else:
            # 标签转换成 one_hot
            cl_onehot = torch.nn.functional.one_hot(cl, num_classes=self.num_item + 1)

            if self.debug == 0:
                print(f"\n\n[debug] before normalize soft target: {soft_target}, max: {soft_target.max()}, 2nd value: {soft_target.kthvalue(soft_target.size(-1) - 1).values.max()}, (soft_target.max(dim=-1).values - soft_target.kthvalue(soft_target.size(-1) - 1).values).max().item() = {(soft_target.max(dim=-1).values - soft_target.kthvalue(soft_target.size(-1) - 1).values).max().item()}")

            soft_target: torch.Tensor
            # soft target logit value varies too much, need to normalize a bit

            diff_max_2nd = (soft_target.max(dim=-1).values - soft_target.kthvalue(soft_target.size(-1) - 1).values).max()

            if diff_max_2nd > 100:
                # soft_target = soft_target / soft_target
                # [a, b] -> [-1, 1]
                # ( x - (a + b) / 2 ) / ((a + b) / 2) = 2 * x / (a + b) - 1
                soft_target = 2 * soft_target / (soft_target.max() + soft_target.min()) - 1
            elif diff_max_2nd > 20:
                soft_target = soft_target / diff_max_2nd

            KL_Loss = nn.functional.kl_div(F.log_softmax(pred, dim=-1), 0.5 * ((soft_target / self.t).softmax(dim=-1) + cl_onehot), reduction='batchmean')

            if self.debug == 0:
                self.debug = 1
                print(f"[debug] soft target {soft_target}, after softmax {soft_target.softmax(dim=-1)}, max in soft {soft_target.softmax(dim=-1).max()}\n\n")

            if ~torch.isinf(KL_Loss):
                tmp_ce = self.ce(pred, cl)
                alpha = self.b
                _al = alpha

                tmp_kl = alpha * KL_Loss

                loss = (1 - alpha) * tmp_ce

                _KL_Loss = KL_Loss.item()
                _CE_Loss = tmp_ce.item()
                _A_CE_Loss = loss.item()
                _A_KL_Loss = tmp_kl.item()

                loss += tmp_kl
            else:
                loss = self.ce(pred, cl)
                _CE_Loss = _A_CE_Loss = loss.item()

        return loss, _CE_Loss, _KL_Loss, _A_CE_Loss, _A_KL_Loss, _al


class CELoss:
    def __init__(self, args):
        self.enable_sample = args.enable_sample
        self.num_item = args.num_items
        self.sample_ratio = args.samples_ratio
        if self.enable_sample:
            self.ce = nn.CrossEntropyLoss()
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.device = args.device

    def compute(self, pred, batch):
        # seqs, labels, rating = batch
        seqs = batch[0]
        labels = batch[1]

        if len(pred.size()) == 3:
            cl = labels[labels > 0]
            pred = pred[labels > 0]
        else:
            cl = labels.squeeze()

        if self.enable_sample:
            raise NotImplementedError("[enable_sample] not fully tested yet")
            # 负样本采样 
            negs = neg_sample(seqs, labels, self.num_item, int(self.num_item * self.sample_ratio))
            negs = torch.LongTensor(list(negs)).repeat(len(cl), 1).to(self.device)
            target = torch.cat((cl.unsqueeze(1), negs), 1)
            # 采样后的one_hot
            one_hot = [1] + [0] * negs.size(-1)
            one_hot = torch.LongTensor(one_hot).repeat(negs.size(0), 1).to(torch.device(self.device))
            # 抽取采样后的结果
            pred = pred.gather(dim=1, index=target)

            # 标签
            label = torch.LongTensor([0] * pred.size(0)).to(torch.device(self.device))
            
            loss = self.ce(pred, label)

        else:
            # 标签转换成 one_hot
            # cl_onehot = torch.nn.functional.one_hot(cl, num_classes=self.num_item + 1)
            loss = self.ce(pred, cl)

        return loss
