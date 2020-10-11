import csv
import numpy as np

# jhuynh

# replace with train, dev, test for each one
test_file = open('../MELD.Raw/dev_sent_emo.csv')
csv_reader = csv.reader(test_file)

test_embeddings = open('../embeddings/dev_embeddings.txt', 'w')

glove_file = open('../glove.840B.300d.txt')
glove_embeddings = {}

for row in glove_file:
    new_row = row.split(" ")
    glove_embeddings[new_row[0]] = np.array(new_row[1:], dtype=float)

# srno_utt : ([glove, glove], # unk)
glove_vectors = {}
glove_avg = []
vocab = []

count = 0
for row in csv_reader:
    if count == 0:
        count+= 1
    else:
        no_unk = 0
        list_embeddings = []

        utt = row[1]
        utt_split = utt.split(" ")

        for word in utt_split:
            if word in glove_embeddings:
                list_embeddings.append(glove_embeddings[word])
                if word not in vocab:
                    vocab.append(word)
                    glove_avg.append(glove_embeddings[word])
            else:
                no_unk += 1

        glove_vectors[row[0]] = (list_embeddings, no_unk)

# calculate unk vector
glove_avg_vec = []
for i in range(300):
    avg = 0
    for j in range(len(glove_avg)):
        avg += glove_avg[j][i]
    glove_avg_vec.append(avg/len(glove_avg))

# calculate avg glove embedding for each sentence
final_glove_embeddings = {}
for key in glove_vectors:
    final_vec = []
    for i in range(300):
        avg = 0
        for j in range(len(glove_vectors[key][0])):
            avg += glove_vectors[key][0][j][i]
        avg += (glove_vectors[key][1] * glove_avg_vec[i])
        final_vec.append(avg/(len(glove_vectors[key][0]) + glove_vectors[key][1]))
    final_glove_embeddings[key] = final_vec
    test_embeddings.write(str(key) + " " + str(final_vec) + "\n")
