import pickle
import csv
import torch
import numpy as np

# use BERT embeddings
pkl_file = '../MELD.Features.Models/features/audio_embeddings_feature_selection_emotion.pkl'
csv_file = '../MELD.Raw/train_sent_emo.csv'
embeddings_final = 'unimodal_audio_train_embeddings.pkl'

# pkl_file = '../MELD.Features.Models/features/audio_embeddings_feature_selection_emotion.pkl'
# csv_file = '../MELD.Raw/dev_sent_emo.csv'
# embeddings_final = 'unimodal_audio_dev_embeddings.pkl'

# pkl_file = '../MELD.Features.Models/features/audio_embeddings_feature_selection_emotion.pkl'
# csv_file = '../MELD.Raw/test_sent_emo.csv'
# embeddings_final = 'unimodal_audio_test_embeddings.pkl'

# missing = ['11_8', '73_0', '97_1', '125_3', '128_0', '128_1', '134_0', '225_11', '254_7', '259_3', '298_6', '300_2', '301_8', '319_2', '354_3', '378_1', '433_1', '465_4', '495_0', '512_0', '512_1', '512_2', '512_3', '517_1', '522_1', '548_5', '573_0', '603_10', '608_0', '608_3', '608_4', '608_5', '608_6', '608_7', '608_8', '638_0', '639_7', '655_0', '700_5', '756_0', '775_0', '810_0', '811_10', '813_1', '818_0', '845_0', '866_0', '936_0', '946_1', '964_15', '967_7', '1009_16', '1016_0', '1021_0', '1024_0']
# missing = ['5_9', '45_10', '70_3', '77_0', '89_0', '104_14', '110_7']
missing = ['2_8', '11_0', '32_3', '49_16', '128_0', '173_6', '175_11', '217_11', '238_4', '268_5', '272_0']

dev_embeddings = pickle.load(open(pkl_file, 'rb'))
dev_csv = open(csv_file)
csv_reader = csv.reader(dev_csv)

convos = []

dev_ids = {}
count = 0
for row in csv_reader:
    if count == 0:
        count += 1
    else:
        emotion = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
        dev_ids[row[5] + '_' + row[6]] = {"dialogue_id": row[5], "utterance_id": row[6], "embedding": dev_embeddings[0][row[5] + "_" + row[6]], "emotion": emotion[row[3]]}
        if row[5] not in convos:
            convos.append(row[5])

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