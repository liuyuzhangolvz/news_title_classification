# news_title_classification

新闻标题文本分类，包含 14 个类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。

评价指标：Accuracy = 分类正确数量 / 需要分类总数量

1. Dataset

数据下载：https://aistudio.baidu.com/aistudio/datasetdetail/103654/0

数据格式

```
  title\tlabel
```

2. Model

hfl/chinese-roberta-wwm-ext + Fully Connected Layer

hfl/chinese-roberta-wwm-ext + TextCNN

3. Train

```
  CUSD_VISIBLE_DEVICES=0 nohup python -u main.py --do_train --output_dir output_16_2e-5 --num_train_epochs 10 > train.log&
```

4. Predict

```
  CUSD_VISIBLE_DEVICES=0 nohup python -u main.py --do_test --output_dir output_16_2e-5 --result_path result_211027.txt > test.log&
 ```
  
5. Reault

Accuracy: 87.53813
