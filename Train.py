from torch import optim
import sklearn.metrics
from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import classification_report

import numpy as np
import torch
from tqdm import tqdm
import time
from Model import ICAT_Classifier
import torch.nn as nn
from Data_preparation import read_dataset, split


def training(loss_function, optimizer, model, num_epochs, train_set, val_set):
    '''
    :param loss_function: the loss function selected for training
    :param optimizer: optimizer selected for training
    :param model: model to train
    :param num_epochs: integer number of epochs for training
    :param train_set: a list of training tuple as (seq,label)
    :param val_set: a list of validation tuple as (seq,label)
    :return:
    '''
    scores, all_loss_val, all_mcc, all_f1, all_acc, all_rec, all_pre = ([] for i in range(7))
    for epoch in range(1, num_epochs + 1):
        all_loss = list()
        start = time.time()
        for i in tqdm(range(len(train_set[0:10]))):
            model.zero_grad()
            sample = train_set[i]
            pred = model(sample[0])
            gold = torch.tensor([sample[1]], dtype=torch.long).cuda()
            loss = loss_function(pred, gold)
            loss.backward()
            all_loss.append(loss.cpu().detach().numpy())
            optimizer.step()
        with torch.no_grad():
            model.eval()
            all_gold = list()
            all_pred = list()
            optimizer.zero_grad()
            for sample in val_set[0:5]:
                pred = model(sample[0])
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
        torch.save(model.state_dict(),"/ICAT_model")


Dataset = 'UniProt/'
p_iden = 60
min_seqs = 10
X, y = read_dataset(Dataset)
X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
train_set = [(X_train[i], y_train[i]) for i in range(len(X_train))]
val_set = [(X_val[i], y_val[i]) for i in range(len(X_val))]
test_set = [(X_test[i], y_test[i]) for i in range(len(X_test))]


model = ICAT_Classifier(12).cuda()
loss_function = nn.CrossEntropyLoss()
learning_rate = 0.000005
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 10
training(loss_function, optimizer, num_epochs, model, train_set, val_set)

with torch.no_grad():
  model.eval()
  all_gold=list()
  all_pred=list()
  for sample in test_set:
    pred = model(sample[0])
    all_gold.append(sample[1])
    pred = np.argmax(pred.cpu().detach().numpy())
    all_pred.append(pred)
with open('./results.txt','w') as f:
    f.write(classification_report(all_gold, all_pred))
