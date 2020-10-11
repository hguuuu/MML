import cv2
import os
import re
import PIL
import torch
import vgg
from matplotlib import pyplot as plt
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from tsnecuda import TSNE
import numpy as np

def process_video(file_path):
    vidcap = cv2.VideoCapture(file_path)
    features = []
    count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        count += 1
        if count % 10 != 1:
            continue
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(PIL.Image.fromarray(image)).to('cuda')
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            features.append(model(input_batch).cpu().numpy())
    return features

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # vid_dir = "/home/shi/Downloads/dev_splits_complete"
    # model = vgg.vgg16().to('cuda')
    # model.eval()
    # df = pd.read_csv("/home/shi/Downloads/dev_sent_emo.csv")
    # labels = []
    # all_features = []
    # for filename in tqdm(list(os.listdir(vid_dir))):
    #     match = re.search("dia(\d+)_utt(\d+).mp4", filename)
    #     if match:
    #         dia = int(match.group(1))
    #         utt = int(match.group(2))
    #         features = process_video(os.path.join(vid_dir, filename))
    #         try:
    #             label = df[(df['Dialogue_ID'] == dia) & (df['Utterance_ID'] == utt)]['Emotion'].values[0]
    #             labels.extend([label] * len(features))
    #             all_features.extend(features)
    #         except Exception as e:
    #             print(e)
    #             continue
    # X = np.array(all_features)
    # labels = np.array(labels)
    # np.save("features.npy", X)
    # np.save("labels.npy", labels)

    X = np.load("features.npy")
    labels = np.load("labels.npy")

    fig, ax = plt.subplots()
    tsne = TSNE(n_components=2, perplexity=15, learning_rate=50, n_iter=5000).fit_transform(X)
    tsne_x = tsne[:, 0]
    tsne_y = tsne[:, 1]
    for l in np.unique(labels):
        i=np.where(labels == l)
        ax.scatter(tsne_x[i], tsne_y[i], label=l)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
