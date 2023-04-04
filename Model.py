import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch import optim
import torch
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import classification_report


class ICAT_Classifier(nn.Module):
    def __init__(self, num_classes):
        '''

        :param num_classes: integer number of classes in the dataset
        '''
        super().__init__()
        self.config = AutoConfig.from_pretrained("Rostlab/prot_bert_bfd")
        self.num_class = num_classes
        self.bert = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input):
        input = self.tokenizer(input, return_tensors="pt", truncation=True, max_length=1024)
        bert_rep = self.bert(input['input_ids'].cuda())
        cls_rep = bert_rep.last_hidden_state[0][0]
        class_scores = self.classifier(cls_rep)
        return F.softmax(class_scores.view(-1, self.num_class), dim=1)

    def trainer(self, num_epochs, train_set, val_set):
        '''
        :param num_epochs: integer number of epochs for training
        :param train_set: a list of training tuple as (seq,label)
        :param val_set: a list of validation tuple as (seq,label)
        :return:
        '''
        loss_function = nn.CrossEntropyLoss()
        learning_rate = 0.000005
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scores, all_loss_val, all_mcc, all_f1, all_acc, all_rec, all_pre = ([] for i in range(7))
        for epoch in range(1, num_epochs + 1):
            all_loss = list()
            start = time.time()
            # training loop
            for i in tqdm(range(len(train_set))):
                optimizer.zero_grad()
                sample = train_set[i]
                pred = self.forward(sample[0])
                gold = torch.tensor([sample[1]], dtype=torch.long).cuda()
                loss = loss_function(pred, gold)
                loss.backward()
                all_loss.append(loss.cpu().detach().numpy())
                optimizer.step()
            # validation loop
            with torch.no_grad():
                self.eval()
                all_gold = list()
                all_pred = list()
                optimizer.zero_grad()
                for sample in val_set[0:5]:
                    pred = self.forward(sample[0])
                    all_gold.append(sample[1])
                    gold = torch.tensor([sample[1]], dtype=torch.long).cuda()
                    loss = loss_function(pred, gold)
                    all_loss_val.append(loss.cpu().detach().numpy())
                    prediction = np.argmax(pred.cpu().detach().numpy())
                    all_pred.append(prediction)
                    if epoch == num_epochs:
                        scores.append((pred.cpu().detach().numpy(), gold.cpu().detach().numpy()))
            print("Avg loss: " + str(np.mean(all_loss)))
            end = time.time()
            print(end - start)
            torch.save(self.state_dict(), "/ICAT_model")


    def test(self, test_set):
        with torch.no_grad():
            self.eval()
            all_gold = list()
            all_pred = list()
            for sample in test_set:
                pred = self.forward(sample[0])
                all_gold.append(sample[1])
                pred = np.argmax(pred.cpu().detach().numpy())
                all_pred.append(pred)
        with open('./results.txt', 'w') as f:
            f.write(classification_report(all_gold, all_pred))







