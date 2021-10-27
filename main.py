import os
import argparse
import random

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import (AdamW, WEIGHTS_NAME, CONFIG_NAME)

from model import BertClassify
from tqdm import tqdm, trange
logger = logging.getLogger(__name__)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    pred_class = np.argmax(preds, axis=1)

    accuracy = (pred_class == labels).mean()
    return {'accuracy': accuracy}



class InputExample(object):
    def __init__(self, title, label=None):
        self.title = title
        self.label = label


class DataProcessor:

    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        file = os.path.join(data_dir, 'train.txt')
        return self._process_data(file)

    def get_valid_examples(self, data_dir):
        logger.info("LOOKING AT {} valid".format(data_dir))
        file = os.path.join(data_dir, 'dev.txt')
        return self._process_data(file)

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        file = os.path.join(data_dir, 'test.txt')
        return self._process_test_data(file)

    def _process_data(self, data_dir):
        examples = []
        max_len_data = 0
        with open(data_dir, 'r') as reader:
            for line in reader:
                data, lable = line.strip().split('\t')
                max_len_data = max(max_len_data, len(data))
                examples.append(InputExample(title=data, label=lable))
        print('The maximum length od data is {}'.format(max_len_data))
        return examples

    def _process_test_data(self, data_dir):
        examples = []
        with open(data_dir, 'r') as reader:
            for line in reader:
                data = line.strip()
                examples.append(InputExample(title=data))
        return examples


class NewsDataset(Dataset):
    def __init__(self, examples, tokenizer, label_list, max_len, with_labels=True):
        self.examples = examples
        self.tokenizer = tokenizer
        self.with_labels = with_labels
        self.max_len = max_len
        self.label2id = {label: idx for idx, label in enumerate(label_list)}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx].title
        label = self.examples[idx].label

        tokenized = self.tokenizer(text,
                                   padding='max_length',  # Pad to max_length
                                   truncation=True,  # Truncate to max_length
                                   max_length=self.max_len,
                                   return_tensors='pt')
        input_ids = tokenized['input_ids']
        token_type_ids = tokenized['token_type_ids']
        attention_mask = tokenized['attention_mask']
        if self.with_labels:
            label_id = self.label2id[label]
            return input_ids, token_type_ids, attention_mask, label_id
        else:
            return input_ids, token_type_ids, attention_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chinese news text title classification',
                                     usage='train.py')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--data_path',
                        default='data',
                        type=str,
                        help='The input data dir, should contain train.txt|valid.txt|test.txt')
    parser.add_argument('--output_dir',
                        default='output',
                        type=str,
                        help='The output dic where the model predictions and checkpoint will be written.')
    parser.add_argument('--result_path',
                        default='reault.txt',
                        type=str,
                        help='The result file in test phrase.')
    parser.add_argument('--model_type',
                        default='bert',
                        type=str)
    parser.add_argument('--pretrained_model',
                        default='hfl/chinese-roberta-wwm-ext',
                        type=str,
                        help='Pre-trained model would be download from Hugging Face.')
    parser.add_argument('--max_len',
                        default=64,
                        type=int,
                        help='The maximum total input sequence length.')
    parser.add_argument('--hidden_size',
                        default=768,
                        type=int,
                        help='Hidden size')
    parser.add_argument('--num_train_epochs',
                        default=10,
                        type=int,
                        help='Total number of train epochs to perform. ')
    parser.add_argument('--gradient_accumulation_steps',
                        default=1,
                        type=int,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='Total batch size.')
    parser.add_argument('--learning_rate',
                        default=2e-5,
                        type=float,
                        help='The learning rate.')
    parser.add_argument('--dropout',
                        default=0.5,
                        type=float,
                        help='The hidden dropout prob.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='random seed for initialization')

    args = parser.parse_args()
    logger.info('args >> {}'.format(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ngpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if ngpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_test:
        raise ValueError("At least one of 'do_train' or 'do_test' must be true. ")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    label_list = ['财经', '彩票', '房产', '股票', '家居', '教育', '科技', '社会', '时尚', '时政', '体育', '星座', '游戏', '娱乐']
    label_size = len(label_list)
    id2label = {idx: label for idx, label in enumerate(label_list)}

    config = BertConfig.from_pretrained(args.pretrained_model)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    albert = BertModel.from_pretrained(args.pretrained_model, config=config)

    train_examples = None
    num_train_optimization_steps = None
    data_processor = DataProcessor()

    model = BertClassify(bert=albert,
                         label_size=label_size,
                         hidden_size=config.hidden_size,
                         hidden_dropout_prob=args.dropout)

    model.to(device)

    global_step = 0
    if args.do_train:
        print('Star to load train dataset...')
        train_examples = data_processor.get_train_examples(args.data_path)
        print('train_examples>>', len(train_examples))
        # print(train_examples[-1].title)
        # print(train_examples[-1].label)
        num_train_optimization_steps = int(
            len(train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        train_dataset = NewsDataset(examples=train_examples,
                                    tokenizer=tokenizer,
                                    label_list=label_list,
                                    max_len=args.max_len)
        #print('train_data[0] >>', train_dataset[0])


        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)

        print('Star to load train dataset...')
        valid_examples = data_processor.get_valid_examples(args.data_path)
        print('dev_examples>>', len(valid_examples))
        valid_dataset = NewsDataset(examples=valid_examples,
                                    tokenizer=tokenizer,
                                    label_list=label_list,
                                    max_len=args.max_len)
        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        for epoch in trange(args.num_train_epochs, desc="Epoch"):
            model.train()
            train_loss = 0
            train_steps = 0
            best_acc = 0.0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                optimizer.zero_grad()
                batch = tuple(p.to(device) for p in batch)
                #print('batch>>', batch)
                inputs = {
                    'input_ids': torch.squeeze(batch[0], dim=1),
                    'attention_mask': torch.squeeze(batch[1], dim=1),
                    'token_type_ids': torch.squeeze(batch[2], dim=1),
                    'labels': batch[3]
                }

                outputs = model(**inputs)
                #print('outputs>>', outputs)
                loss = outputs[1]

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                train_loss += loss.detach().item()
                train_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    loss.backward()
                    optimizer.step()

            print('Start to eval...')
            model.eval()
            eval_loss = 0
            eval_steps = 0
            preds = None
            for batch in tqdm(valid_dataloader, desc="Evaluating"):
                batch = tuple(p.to(device) for p in batch)

                with torch.no_grad():
                    inputs = {
                        'input_ids': torch.squeeze(batch[0], dim=1),
                        'attention_mask': torch.squeeze(batch[1], dim=1),
                        'token_type_ids': torch.squeeze(batch[2], dim=1),
                        'labels': batch[3]
                    }
                    outputs = model(**inputs)
                    logits, tmp_eval_loss = outputs

                    eval_loss += tmp_eval_loss.detach().mean().item()

                eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = batch[3].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / eval_steps

            result = compute_metrics(preds, out_label_ids)
            loss = train_loss / train_steps if args.do_train else None

            result['eval_loss'] = eval_loss
            result['global_step'] = global_step
            result['loss'] = loss


            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval result *****")
                for key in sorted(result.keys()):
                    print("%s = %s" % (key, str(result[key])))
                    logger.info("%s = %s", key, str(result[key]))
                    writer.write("%s = %s \n" % (key, str(result[key])))
            if result['accuracy'] > best_acc:
                best_acc = result['accuracy']
                output_model_file = os.path.join(args.output_dir,
                                                 "best_model_" + str(args.pretrained_model).split('/')[-1] + WEIGHTS_NAME)
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                torch.save(model.state_dict(), output_model_file)
                # model.config.to_json_file(output_config_file)
                # tokenizer.save_vocabulary(args.output_dir)

    if args.do_test:
        if not args.result_path:
            raise ValueError("The result path must be choose.")
        if os.path.exists(args.result_path):
            raise ValueError("Result file already exists.")
        print('Star to load test dataset...')
        test_examples = data_processor.get_test_examples(args.data_path)
        print('dev_examples>>', len(test_examples))
        test_dataset = NewsDataset(examples=test_examples,
                                   tokenizer=tokenizer,
                                   label_list=label_list,
                                   max_len=args.max_len,
                                   with_labels=False)
        test_dataloader = DataLoader(test_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False)

        print('Loading model...')
        output_model_file = os.path.join(args.output_dir,
                                         "best_model_" + str(args.pretrained_model).split('/')[-1] + WEIGHTS_NAME)
        checkpoint = torch.load(output_model_file)
        #print('checkpoint>>', checkpoint)
        model.load_state_dict(checkpoint)
        print('Start to test...')
        model.eval()
        preds = None
        for batch in tqdm(test_dataloader, desc="Testing"):
            batch = tuple(p.to(device) for p in batch)

            with torch.no_grad():
                inputs = {
                    'input_ids': torch.squeeze(batch[0], dim=1),
                    'attention_mask': torch.squeeze(batch[1], dim=1),
                    'token_type_ids': torch.squeeze(batch[2], dim=1)
                    # 'labels': batch[3]
                }
                logits = model(**inputs)
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        assert len(test_examples) == len(preds)
        pred_labels = np.argmax(preds, axis=1)
        print('pred_labels size>>', len(pred_labels))

        with open(args.result_path, 'w') as writer:
            for label_id in pred_labels:
                writer.writelines(id2label[label_id] + '\n')
        print('Test result saved to {}'.format(result_path))
