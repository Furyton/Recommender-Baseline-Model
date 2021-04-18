# usage

[中文](./README_CN.md)
## warning

remember to comment [this line](https://github.com/Furyton/Recommender-Baseline-Model/blob/7232e7f2033e28c1c4ce75bf7087bd066924edc1/NerualNetwork/sas4rec%20only/main.py#L30) if you are using slurm.

## sas4rec

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
dataset: dataset file name, must be text format, e.g. "ml.txt"

train_dir: where you put the log and model information in the directory, e.g. "default"

batch_size: batch size for training and test, e.g. 128

lr: learning rate, e.g. 0.001

maxlen: Length of the input sequence, e.g. 200

hidden_units: size of the hidden layers, e.g. 64

num_blocks: number of transformer layers, e.g. 2

num_epochs: number of epoch, e.g. 200

num_heads: number of heads for multi-attention, e.g. 2

dropout_rate: dropout rate in the sas model, e.g. 0.2

l2_emb: normalization, e.g. 0.0

device: e.g. "cpu"

inference_only: true or false, default: false

state_dict_path: the path of the model weight dict to load
```

this is the config you can edit for `sas only`.

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
python main.py --inference_only=true
```
