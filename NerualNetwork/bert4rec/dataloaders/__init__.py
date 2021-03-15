from collections import defaultdict

from .bert import BertDataloader

DATALOADERS = {
    BertDataloader.code(): BertDataloader,
}


def data_partition(fname, max_len, prop_sliding_window):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = []
    user_valid = []
    user_test = []
    # assume user/item index starting from 1
    f = open(fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    data = []

    sliding_step = int(prop_sliding_window * max_len) if prop_sliding_window != -1.0 else max_len

    for user in User:
        if len(User[user]) < 3:
            continue
        if len(User[user]) <= max_len:
            data.append(User[user])
        else:
            # print("FILM LOVER: ", user)
            beg_idx = range(len(User[user]) - max_len, 0, -sliding_step)
            # beg_idx.append(0)
            for i in beg_idx[::-1]:
                data.append(User[user][i:i + max_len])

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



def dataloader_factory(args):
    dataset = data_partition(args.data_dir, args.bert_max_len, args.prop_sliding_window)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
