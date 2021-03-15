def set_template(args):
    args.mode = 'train'

    args.split = 'leave_one_out'
    args.dataloader_code = 'bert'
    batch = 64
    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch

    args.train_negative_sampler_code = 'popular'
    args.train_negative_sample_size = 0
    args.train_negative_sampling_seed = 0
    args.test_negative_sampler_code = 'popular'
    args.test_negative_sample_size = 100
    args.test_negative_sampling_seed = 98765
    args.prop_sliding_window = 0.3

    args.trainer_code = 'bert'
    args.device = 'cpu'
    args.num_gpu = 1
    args.device_idx = '0'
    args.optimizer = 'Adam'
    args.lr = 0.0001
    args.decay_step = 25
    args.gamma = 1.0
    args.num_epochs = 500
    args.metric_ks = [1, 5, 10, 20]
    args.best_metric = 'NDCG@10'

    args.model_code = 'bert'
    args.model_init_seed = 0

    args.bert_dropout = 0.25
    args.bert_hidden_dropout = 0.25

    args.data_dir = 'ml.txt'

    args.path = None

    args.bert_hidden_units = 64
    args.bert_mask_prob = 0.3
    args.bert_max_len = 200
    args.bert_num_blocks = 2
    args.bert_num_heads = 2

