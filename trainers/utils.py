import torch


def recall(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / labels.sum(1).float()).mean().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(n, k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()

def _cal_metrics(labels, scores, metrics, ks, star=''):
    answer_count = labels.sum(1).to(torch.int)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall' + star + '@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float()).to(scores.device)
        dcg = (hits * weights).sum(1)
        idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count]).to(scores.device)
        ndcg = (dcg / idcg).mean().item()
        metrics['NDCG' + star + '@%d' % k] = ndcg

        position_mrr = torch.arange(1, k + 1).to(scores.device)
        weights_mrr = 1 / position_mrr.float()
        mrr = (hits * weights_mrr).sum(1)
        mrr = mrr.mean().item()

        metrics['MRR' + star + '@%d' % k] = mrr


# B x C, B x C
def recalls_ndcgs_and_mrr_for_ks(scores, labels, ks, ratings):
    metrics = {}

    _cal_metrics(labels, scores, metrics, ks)

    # ratings = ratings.squeeze()
    # scores = scores[ratings == 1.]
    # labels = labels[ratings == 1.]

    # _cal_metrics(labels, scores, metrics, ks, '*')

    return metrics
