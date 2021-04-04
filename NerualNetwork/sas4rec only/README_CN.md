# 使用方法

[English](./README.md)
## 注意

如果你使用slurm来调度你的gpu资源，需要注释掉[这一行](https://github.com/Furyton/Recommender-Baseline-Model/blob/7232e7f2033e28c1c4ce75bf7087bd066924edc1/NerualNetwork/sas4rec%20only/main.py#L30)

## sas4rec

### 数据集格式

数据集放在了`Data` 目录下，文本格式，比如 `ml.txt`

user和item下标从 1 开始，必须连续

每行代表一个交互：user_id item_id

对于同一个用户，物品出现顺序不能打乱，但不同用户的数据出现位置可以混在一起

sample
```
1 2
1 5
1 1
2 1
2 2
1 7
```

### 配置
需要更改配置文件 `config.json`


```
dataset: dataset file name, must be text format, e.g. "ml.txt"

train_dir: 用来存放模型信息结果等的目录, e.g. "default"

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

### 训练

#### Step 1
将数据集放到 `Data` 目录下，确保格式正确

#### Step 2
编辑 `config.json` 文件

#### Step 3
安装库

```
pip install -r requirements.txt
```

#### Step 4
run
```
python main.py
```

### note

可以使用命令行参数，它会覆盖 `config.json` 

example

```
python main.py --inference_only=true
```

命令行参数中可以选择配置文件


example

```
python main.py --config_file=config1.json
```
