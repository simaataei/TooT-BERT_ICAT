
from Model import ICAT_Classifier
from Data_preparation import read_dataset, split




Dataset = "UniProt/"
p_iden = 60
min_seqs = 10
X, y = read_dataset(Dataset)
X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)
train_set = [(X_train[i], y_train[i]) for i in range(len(X_train))]
val_set = [(X_val[i], y_val[i]) for i in range(len(X_val))]
test_set = [(X_test[i], y_test[i]) for i in range(len(X_test))]


model = ICAT_Classifier(12).cuda()
num_epochs = 10
model.trainer(num_epochs, train_set, val_set)
model.test(test_set)
