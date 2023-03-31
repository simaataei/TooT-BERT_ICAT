from sklearn.model_selection import train_test_split

data_label_path = 'Dataset/Data_label_pairs/'


def read_dataset(Dataset, p_iden, min_seqs):
    '''

    :param Dataset: Dataset name as str from two options of 'UniProt', 'SwissProt'
    :param p_iden: Percent identity of sequences in the dataset as int
    :param min_seqs: Minimum number of sequences in each class of the dataset from option 3 or 10 as int
    :return: X, y as 2 lists of sequences and the labels in the dataset
    '''

    X = []
    y = []
    with open(data_label_path + Dataset + Dataset +'_ICAT_' + str(p_iden)+'_t'+str(min_seqs)+'_accession_label.txt' ) as f:
        data = f.readlines()
        for d in data:
            X.append(d.strip('\n').split(',')[0])
            y.append(d.strip('\n').split(',')[1])
    return X,y

def split(X,y):
    '''

    :param X: a list of sequences of the dataset
    :param y: a list of labels for each sequence
    :return: X_train, X_val, X_test, y_train, y_val, y_test as training, validation and test sets
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    return X_train, X_val, X_test, y_train, y_val, y_test

