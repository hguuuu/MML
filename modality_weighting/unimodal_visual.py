import pickle
import csv
import torch
import numpy as np

# use 30 frame embeddings
pkl_file = '../MML/vis_features_openface/train_vis_30_frames_aus.pickle'
csv_file = '../MELD.Raw/train_sent_emo.csv'
embeddings_final = 'unimodal_visual_train_embeddings.pkl'

# pkl_file = '../MML/vis_features_openface/dev_vis_30_frames_aus.pickle'
# csv_file = '../MELD.Raw/dev_sent_emo.csv'
# embeddings_final = 'unimodal_visual_dev_embeddings.pkl'

# pkl_file = '../MML/vis_features_openface/test_vis_30_frames_aus.pickle'
# csv_file = '../MELD.Raw/test_sent_emo.csv'
# embeddings_final = 'unimodal_visual_test_embeddings.pkl'

dev_embeddings = pickle.load(open(pkl_file, 'rb'))
dev_csv = open(csv_file)
csv_reader = csv.reader(dev_csv)

convos = []

missing = []

dev_ids = {}
count = 0
for row in csv_reader:
    if count == 0:
        count += 1
    else:
        try:
        # print(dev_embeddings[row[5] + "_" + row[6]])
            new_embed = torch.from_numpy(dev_embeddings[row[5] + "_" + row[6]])
            new_embed = torch.transpose(new_embed, 0, 1)
            new_embed = torch.flatten(new_embed).view(1, -1)
            emotion = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
            dev_ids[row[5] + '_' + row[6]] = {"dialogue_id": row[5], "utterance_id": row[6], "embedding": new_embed, "emotion": emotion[row[3]]}
            if row[5] not in convos:
                convos.append(row[5])
        except:
            missing.append(row[5] + '_' + row[6])

# print(len(dev_ids['2_7']['embedding']))

new_dev_ids = {}
# concatenate embeddings for context (2 before)
count = 0
for key in dev_ids:
    dia_id = key.split('_')[0]
    utt_id = key.split('_')[1]
    if int(utt_id) > 1:
        try:
            new_embedding = torch.cat((torch.from_numpy(np.asarray(dev_ids[dia_id + '_' + str(int(utt_id) - 2)]['embedding'])).to(torch.double), torch.from_numpy(np.asarray(dev_ids[dia_id + '_' + str(int(utt_id) - 1)]['embedding'])).to(torch.double), torch.from_numpy(np.asarray(dev_ids[key]['embedding'])).to(torch.double)))
            new_dev_ids[count] = {"dialogue_id": dia_id, "utterance_id": utt_id, "embedding": new_embedding, "emotion": dev_ids[key]["emotion"]}
            count += 1
        except KeyError:
            pass

pickle.dump(new_dev_ids, open(embeddings_final, "wb"))