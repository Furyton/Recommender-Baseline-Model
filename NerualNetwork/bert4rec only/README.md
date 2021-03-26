# usage

## bert4rec only

### dataset format

the datasets are stored in the `Data` directory in text format, e.g. `ml.txt`

the index of users and items starts at 1, and the id should be both continuous.

each line indicates an interaction: user_id item_id

for any single user, the order of the interacted items should be preserved. But the order between different users could be mixed.

example
```
1 2
1 5
1 1
2 1
2 2
1 7
```

### configuration
the config file `config.json` is all you need to change

```
mode: train or test, e.g. "train"
load_processed_dataset: the dataset you put in the `Data` will be processed into .pkl, 
                        you can load it for saving time. true or false, e.g. false
processed_dataset_path: absolute path, e.g. "C:processed/ml.pkl"


test_model_path: absolute path, e.g. "C:model/my_model.pth"
resume_path: absolute path, e.g. "C:model/check_point_model.pth"


dataloader_code: bert(this is bert only), e.g. "bert"
dataloader_random_seed: float, default=0.0


train_batch_size: batch size, e.g. 64
val_batch_size: batch size, e.g. 64
test_batch_size: batch size, e.g. 64


prop_sliding_window: propotion of the sliding window step, if the input seq is exceeding the max_len, 
                     you can use a sliding window to generate a sequence of input. default: 0.1,  
                     if you don't want this sliding windown, set this parameter as -1.0.

worker_number: for multi-processer, usually it could be 4 times \#cpu core you have


train_negative_sampler_code: popular or random, e.g. "popular"
train_negative_sample_size: for bert, this is unused, set as 0
train_negative_sampling_seed: 0


test_negative_sampler_code: popular or random, e.g. "popular"
test_negative_sample_size: default: 100
test_negative_sampling_seed: 0


trainer_code: bert
device: cpu or cuda, default: "cpu"
num_gpu: default: 1
device_idx: note: this is a string, you can type "0", or "0, 1, 2"


optimizer: "Adam" or "SGD"
lr: learning rate, e.g. 0.001
weight_decay: l2 regularization, e.g. 0.01

decay_step: decay step for StepLR, e.g. 15
gamma: Gamma for StepLR, e.g. 0.1


num_epochs: number of epochs for training, e.g. 100

log_period_as_iter: after every certain number of iterations, the model's weight will be saved as checkpoint

metric_ks: ks for Metric@k, there are 3 types of metric: MRR, NDCG and HIT, e.g. [10, 20, 50]
best_metric: Metric for determining the best model, e.g. "NDCG@10"

show_process_bar: show the processing bar when training or testing, true or false, e.g.false

model_code: bert

model_init_seed



bert_max_len: Length of sequence for bert, e.g. 200
bert_hidden_units: Size of hidden vectors, e.g. 64
bert_num_blocks: number of transformer layers, e.g. 2
bert_num_heads: number of heads for multi-attention, e.g. 2
bert_dropout: used for transformer blocks
bert_hidden_dropout: used for hidden unit layers


experiment_dir: where you want to put you trained model and log info in, e.g. "experiments"
experiment_description: default: "test"
dataset_name: the dataset filename you put in the `Data` dir, e.g. "ml.txt"
```

this is the config you can edit for `bert only`.

### training

#### Step 1

put your dataset into the 'Data' dir, make sure the format meet the requirement.

#### Step 2
edit the `config.json` file

#### Step 3

install Essential Package.
```
pip install -r requirements.txt
```

#### Step 4

run
```
python main.py
```

### note
you can also config the parameters in the command line. they will overwrite the `config.json`

example

```
python main.py --mode=test
```
