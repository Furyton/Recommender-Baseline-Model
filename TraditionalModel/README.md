# Usage

## dataset format

text format

each line indicates a sequence for an user

the user_id and item_id start at 1, must be continuous

example
```
2	[5131, 29111, 3340, 171, 1879, 8756]
3	[10025, 29115, 17046, 597, 5666, 10026, 10027, 17047]
```

## parameter

```
model_type: bpr, knn, pop

train_dir: training dataset directory, e.g. "data/pop_test.txt"

k : for KNN model, default 100

epoch : for bpr model, default 500

device: for bpr model training, cpu or cuda , default cpu
device_num : for cuda device id , default 0
lr : learning rate for bpr model training
hidden_units : bpr hidden unit number, default 64
model_dir : directory to save the bpr model weitgh dict, for resume training

candicate_num : candidate number for evaluating, default 100
```

### training

#### step 1
preprocess your dataset in the correct format

#### step 2
install package

```
pip install -r requirements.txt
```

#### step 3
run

for knn model

```
python main.py --model_type=knn --dataset_path=./data.txt --k=100 --candidate_num=100
```

for pop model


```
python main.py --model_type=pop --dataset_path=./data.txt --candidate_num=100
```


for bpr model


```
python main.py --model_type=bpr --dataset_path=./data.txt --epoch=200 --device=cpu --lr=0.001 --hidden_units=64 --candidate_num=100
```