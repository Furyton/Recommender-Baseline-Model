# Neural Network
The Neural Network Model includes Bert4Rec and SAS4Rec

the majority of the code in bert4rec are modified from [Jaywonchung](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch) , and [pmixer](https://github.com/pmixer/SASRec.pytorch) for SAS4rec

both models are implemented in pytorch


```
- sas4rec only
- bert4rec&sas4rec
```


note: I combine these two models into the same framework('bert4rec&sas4rec'), which means you can choose the model you'd like to use by typing in the parameter. The efficiency may be lost due to the generosity, if there is any problem when using the SASrec  you can turn to use the model in 'sas4rec only' or you can commit an issue here.

## usage

### bert4rec and sas4rec
using the same training framework

[Click here to check out the usage for bert4rec and sas4rec](bert4rec%20only/README.md)


### sas4rec
modified from the original [pmixer](https://github.com/pmixer/SASRec.pytorch)

[Click here to check out the usage for sas4rec](sas4rec%20only/README.md)
