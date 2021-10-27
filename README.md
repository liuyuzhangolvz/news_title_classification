# news_title_classification

1. Dataset

数据下载：https://aistudio.baidu.com/aistudio/datasetdetail/103654/0

2. Train

```
  CUSD_VISIBLE_DEVICES=0 nohup python -u main.py --do_train --output_dir output_16_2e-5 --num_train_epochs 10 > train.log&
```

3. Predict

```
  CUSD_VISIBLE_DEVICES=0 nohup python -u main.py --do_test --output_dir output_16_2e-5 --result_path result_211027.txt > test.log&
 ```
  
