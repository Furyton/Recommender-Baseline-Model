# Recommender-Baseline-Model
common used Recommend System Baseline Model, including the traditional statistical model, and the Neural Network Model. Focus on the SRS (Sequential Recommend System).

## info

### Neural Network

The Neural Network Model includes Bert4Rec and SAS4Rec

the majority of the code in bert4rec is from [Jaywonchung](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch) , and [pmixer](https://github.com/pmixer/SASRec.pytorch) for SAS4rec

both models are implemented in pytorch



```
- bert4rec only
- sas4rec only
- bert4rec&sas4rec
```



note: I combine these two models into the same framework('bert4rec&sas4rec'), which means you can choose the model you'd like to use by typing in the parameter. The efficiency may be lost due to the generosity, if there is any problem when using the SASrec  you can turn to use the model in 'sas4rec only' or you can commit an issue here.



## usage

### bert4rec only

#### dataset format

the dataset are stored in the `Data` directory in text format



the config file `config.json` is 



