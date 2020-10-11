from transformers import AutoTokenizer, BertModel
import csv
import torch
import pickle

# jhuynh
# https://huggingface.co/transformers/quickstart.html
# https://huggingface.co/transformers/model_doc/bert.html#bertmodel
# https://docs.python.org/3/library/pickle.html#:~:text=%E2%80%9CPickling%E2%80%9D%20is%20the%20process%20whereby,back%20into%20an%20object%20hierarchy.

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# replace with dev, train, test for each one
test_file = open('../MELD.Raw/dev_sent_emo.csv')
csv_reader = csv.reader(test_file)

test_embeddings = open('../embeddings/dev_embeddings_bert.pkl', 'wb')

model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

embeddings = {}
count = 0
with torch.no_grad():
    for row in csv_reader:
        if count == 0:
            count+= 1
        else:
            tokens = tokenizer(row[1].lower(), return_tensors="pt")['input_ids']
            embeddings[row[0]] = model(tokens)[0]

pickle.dump(embeddings, test_embeddings)
