import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig



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







