from pathlib import Path
import pickle
from dataloaders.MaskDataloader import MaskDataloader
from dataloaders.NextItemDataloader import NextItemDataloader

from os import path

import pandas as pd
from sklearn.utils import shuffle

from config import *

DATALOADERS = {
    MaskDataloader.code(): MaskDataloader,
    NextItemDataloader.code(): NextItemDataloader,
}

def reindex(df: pd.DataFrame, columns:list):
    print("[INFO: reindex]")
    for column in columns:
        df.loc[:, column] = df[column].map(dict(zip(shuffle(df[column].unique()), range(1, len(df[column].unique())+1))))

def drop_cold(df: pd.DataFrame, min_user: int, min_item: int):
    print(f'\n=====\n[INFO: drop_cold] filtering out cold users (who interacts with less or eq than {min_user} items) and cold items (which is interacted by less or eq than {min_item} users)')

    print(f"[drop_cold] before filtering, there are {len(df[SESSION_ID].unique())} users, {len(df[ITEM_ID].unique())} items ")

    max_iter = 10

    while True:
        max_iter -= 1
        if max_iter <= 0:
            raise RecursionError("[drop_cold] iterated too many times (10 times). please consider another denser dataset.")
        
        user_cnt = df.groupby(SESSION_ID).count()
        cold_user_id = user_cnt[user_cnt[RATING] < min_user].index

        item_cnt = df.groupby(ITEM_ID).count()
        cold_item_id = item_cnt[item_cnt[RATING] < min_item].index

        if len(cold_user_id) == 0 and len(cold_item_id) == 0:
            print(f"[drop_cold] after {10 - max_iter - 2} filterings, there are {len(df[SESSION_ID].unique())} users, {len(df[ITEM_ID].unique())} items")

            print(f"[drop_cold] user desc")

            print(user_cnt.drop(columns=ITEM_ID).rename(columns={RATING: "seq_len"}).describe())

            print(f"[drop_cold] item desc")

            print(item_cnt.drop(columns=SESSION_ID).rename(columns={RATING: "item_pop"}).describe())

            return df.copy()
        
        if len(cold_user_id) > 0:
            df = df.drop(index=df[df[SESSION_ID].isin(cold_user_id)].index).reset_index(drop=True).copy()
        
        if len(cold_item_id) > 0:
            df = df.drop(index=df[df[ITEM_ID].isin(cold_item_id)].index).reset_index(drop=True).copy()
        
        reindex(df, [SESSION_ID])

def df_data_partition(args, dataframe: pd.DataFrame, use_rating, do_reindex=True, pd_itemnum=None, only_good=False) -> list:
    max_len, prop_sliding_window = args.max_len, args.prop_sliding_window

    if not dataframe.columns.isin([SESSION_ID, ITEM_ID, RATING]).all():
        raise ValueError

    if only_good:
        print("[df_data_partition]: filtering out items user dislike.")
        dataframe = dataframe[dataframe[RATING]==1].copy()
    
    dataframe = drop_cold(dataframe, args.min_length, args.min_item_inter)

    if do_reindex:
        reindex(dataframe, [SESSION_ID, ITEM_ID])
    else:
        reindex(dataframe, [SESSION_ID])

    print('\n==========\n[df_data_partition] dataset summary:')
    print(dataframe.describe())

    print('==========')
    
    # sliding_step = int(prop_sliding_window * max_len) if prop_sliding_window != -1.0 else max_len
    sliding_step = 20

    def process(df: pd.DataFrame, column, data):
        s = df.loc[:, column]
        if len(s) <= max_len:
            data.append(list(s.to_numpy()))
        else:
            beg_idx = range(len(s) - max_len, 0, -sliding_step)

            if beg_idx[0] != 0:
                data.append(list(s.iloc[0:max_len].to_numpy()))

            for i in beg_idx[::-1]:
                data.append(list(s.iloc[i:i+max_len].to_numpy()))


    sessoin_group = dataframe.groupby(SESSION_ID)

    item_data = []

    sessoin_group.apply(lambda x: process(x, ITEM_ID, item_data))

    itemnum = get_itemnum(dataframe) if pd_itemnum is None else pd_itemnum
    args.num_items = itemnum

    usernum = len(item_data)

    item_train, item_valid, item_test = [], [], []

    for seq in item_data:
        item_train.append(seq[:-2])
        item_valid.append([seq[-2]])
        item_test.append([seq[-1]])

    if use_rating is True:
        rating_data = []
        rating_train, rating_valid, rating_test = [], [], []
        sessoin_group.apply(lambda x: process(x, RATING, rating_data))

        for seq in rating_data:
            rating_train.append(seq[:-2])
            rating_valid.append([seq[-2]])
            rating_test.append([seq[-1]])

        return [item_train, item_valid, item_test, usernum, itemnum, rating_train, rating_valid, rating_test]
    else:
        return [item_train, item_valid, item_test, usernum, itemnum]

def get_itemnum(dataframe: pd.DataFrame) -> int:
    if not dataframe.columns.isin([SESSION_ID, ITEM_ID, RATING]).all():
        raise ValueError
    itemnum = len(dataframe[ITEM_ID].unique())

    return itemnum

# header contains dataset_name, min_user, min_item, good only?, do_reindex?, use_rating? 
def check_dataset_cache(args, header) -> bool:
    print('[check_dataset_cache] check if the cache is generated under this configuration')

    def warning_report(field_name, field1, field2):
        print(f'[WARNING check_dataset_cache] {field_name}: {field1} and {field2} maybe different configurations? I refuse to use this cache.')

    if args.dataset_name != header['dataset_name']:
        warning_report('dataset name', args.dataset_name, header['dataset_name'])
        return False
    if args.min_length != header['min_user']:
        warning_report('min length', args.min_length, header['min_user'])
        return False
    if args.min_item_inter != header['min_item']:
        warning_report('min item interaction count', args.min_item_inter, header['min_item'])
        return False
    if args.good_only != header['good_only']:
        warning_report('good only', args.good_only, header['good_only'])
        return False
    if args.do_reindex != header['do_reindex']:
        warning_report('do reindex', args.do_reindex, header['do_reindex'])
        return False
    if args.use_rating != header['use_rating']:
        warning_report('use rating', args.use_rating, header['use_rating'])
        return False
    
    print('[check_dataset_cache] correct.')
    return True

def gen_dataset(args) -> list:
    print(f'[gen_dataset] processing dataset {args.dataset_name}')

    current_directory = path.dirname(__file__)
    parent_directory = path.split(current_directory)[0]
    dataset_filepath = path.join(parent_directory, RAW_DATASET_ROOT_FOLDER, args.dataset_name)
    data = pd.read_csv(dataset_filepath)
    dataset = df_data_partition(args, data, use_rating=args.use_rating, do_reindex=args.do_reindex, pd_itemnum=args.num_items, only_good=args.good_only)

    args.num_items = dataset[4]

    return dataset

def gen_cache_path(args) -> Path:
    current_directory = path.dirname(__file__)
    parent_directory = path.split(current_directory)[0]

    cache_filename = args.dataset_cache_filename or '{}-{}-{}.pkl'.format(args.dataset_name.split('.')[0], args.min_length, args.min_item_inter)

    folder = Path(parent_directory).joinpath(RAW_DATASET_ROOT_FOLDER, PROCESSED_DATASET_CACHE_FOLDER)

    if not folder.exists():
        folder.mkdir()

    filename = folder.joinpath(cache_filename)

    return filename

# cache processed dataset

def cache_dataset(args, dataset):
    header = {  'dataset_name': args.dataset_name, 
                'min_user':     args.min_length, 
                'min_item':     args.min_item_inter,
                'good_only':    args.good_only,
                'do_reindex':   args.do_reindex,
                'use_rating':   args.use_rating}

    cache_path = gen_cache_path(args)

    with cache_path.open('wb') as f:
        pickle.dump((header, dataset), f)

def dataloader_factory(args, export_root):
    if args.load_processed_dataset:
        cache_file_path = gen_cache_path(args)

        if not cache_file_path.exists():
            print('[dataloader_factory] cache file not found. regenerating')

            dataset = gen_dataset(args)

            cache_dataset(args, dataset)
        else:
            if cache_file_path.is_file():
                print(f"[dataloader_factory] loading processed dataset cache in {cache_file_path}")
                dataset_cache = pickle.load(cache_file_path.open('rb'))

                header, dataset = dataset_cache

                args.num_items = dataset[4]

                if not check_dataset_cache(args, header):
                    print('[dataloader_factory] bad cache detected. regenerating')

                    dataset = gen_dataset(args)

                    cache_dataset(args, dataset)
            else:
                raise ValueError("[dataloader_factory] not a file")
    elif args.save_processed_dataset:
        dataset = gen_dataset(args)

        cache_dataset(args, dataset)
    else:
        dataset = gen_dataset(args)

    dataloader_ = DATALOADERS[args.dataloader_type]

    dataloader = dataloader_(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()

    return train, val, test, dataset
