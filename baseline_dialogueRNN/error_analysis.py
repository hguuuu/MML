import pandas
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# https://seaborn.pydata.org/examples/spreadsheet_heatmap.html

preds = pandas.read_csv("prediction_emotion_output.csv")

# % wrong for each emotion

def emotion_wrong(df, emotion):
    whole = list(df[df['Emotion'] == emotion]['Utterance'])
    arr = list(df[df['Emotion'] == emotion]['Pred_Emotion'])
    other = defaultdict(int)

    for i in range(len(arr)):
        other[arr[i]] += 1
            # get examples for whatever emotion you want
        # if arr[i] == emotion:
        #     if emotion=='disgust':
        #         print('correct', whole[i])
        # else:
        #     if emotion=='disgust':
        #         print('incorrect', whole[i])

    for key in other:
        other[key] = other[key]/len(arr)

    return other

emotions = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
final_arr = []
for emotion in emotions:
    oth = emotion_wrong(preds, emotion)
    arr = []
    for emo in emotions:
        arr.append(oth[emo])
    final_arr.append(arr)

f, ax = plt.subplots()
sns.heatmap(final_arr, annot=True, ax=ax, xticklabels=emotions, yticklabels=emotions)
f.savefig('fig.png')

# % wrong for each sentiment

sentiment = ["neutral", "positive", "negative"]
neg_sentiment = ["surprise", "fear", "sadness", "disgust", "anger"]

def sentiment_wrong(emotion):
    whole = list(preds[preds['Emotion'] == emotion]['Utterance'])
    arr = list(preds[preds['Emotion'] == emotion]['Pred_Emotion'])
    other = defaultdict(int)

    for i in range(len(arr)):
        if arr[i] in neg_sentiment:
            other["negative"] += 1
        elif arr[i] == "neutral":
            # if emotion == "joy":
            #     print("joy", whole[i])
            # elif emotion in neg_sentiment:
            #     print('negative', whole[i])
            other["neutral"] += 1
        else:
            other["positive"] += 1

    return other

final_arr = []
neg_emotions = np.zeros(3)
for emotion in neg_sentiment:
    oth = sentiment_wrong(emotion)
    neg_emotions[0] += oth["neutral"]
    neg_emotions[1] += oth["positive"]
    neg_emotions[2] += oth["negative"]
    
x = sentiment_wrong("neutral")["neutral"] + sentiment_wrong("neutral")["positive"] + sentiment_wrong("neutral")["negative"]
y = sentiment_wrong("joy")["neutral"] + sentiment_wrong("joy")["positive"] + sentiment_wrong("joy")["negative"]
z = neg_emotions[0] + neg_emotions[1] + neg_emotions[2]
final_arr.append([sentiment_wrong("neutral")["neutral"]/x, sentiment_wrong("neutral")["positive"]/x, sentiment_wrong("neutral")["negative"]/x])
final_arr.append([sentiment_wrong("joy")["neutral"]/y, sentiment_wrong("joy")["positive"]/y, sentiment_wrong("joy")["negative"]/y])
final_arr.append([neg_emotions[0]/z, neg_emotions[1]/z, neg_emotions[2]/z])

f, ax = plt.subplots()
sns.heatmap(final_arr, annot=True, ax=ax, xticklabels=sentiment, yticklabels=sentiment)
f.savefig('fig1.png')

# % wrong for time bins, <2 seconds, <10 seconds, > 10
timebins = []

for iteration in preds.iterrows():
    start = float(str.replace(iteration[1].tolist()[-3], ',', '.').split(":")[1])*60 + float(str.replace(iteration[1].tolist()[-3], ',', '.').split(":")[2])
    end = float(str.replace(iteration[1].tolist()[-2], ',', '.').split(":")[1])*60 + float(str.replace(iteration[1].tolist()[-2], ',', '.').split(":")[2])
    timebins.append(end-start)

new_preds = preds.copy()
new_preds.insert(11, 'time_diff', timebins)

def time_wrong_calc(df, emotion):
    whole = list(df[df['Emotion'] == emotion]['Utterance'])
    arr = list(df[df['Emotion'] == emotion]['Pred_Emotion'])
    other = defaultdict(int)

    for i in range(len(arr)):
        other[arr[i]] += 1
        if arr[i] == emotion:
            # get examples for whatever emotion you want
            if emotion=='sadness':
                print('correct', whole[i])
        else:
            if emotion=='sadness':
                print('incorrect', whole[i])
                print(arr[i])

    return other

def time_wrong(time_low, time_high, fig):
    counts = defaultdict(int)
    whole = list(new_preds[(new_preds['time_diff'] < time_high) & (new_preds['time_diff'] > time_low)]['Utterance'])

    arr = new_preds[(new_preds['time_diff'] < time_high) & (new_preds['time_diff'] > time_low)]

    emotions = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
    final_arr = []
    for emotion in emotions:
        oth = time_wrong_calc(arr, emotion)
        arr2 = []
        ct = 0
        for emo in emotions:
            arr2.append(oth[emo])
            ct += oth[emo]
        final_arr.append(arr2)
        counts[emotion] = ct

    print(counts)

    f, ax = plt.subplots()
    sns.heatmap(final_arr, annot=True, ax=ax, xticklabels=emotions, yticklabels=emotions)
    f.savefig(fig + '.png')

# time_wrong(0, 2, 'fig2')
# time_wrong(2, 6, 'fig3')
time_wrong(6, 100, 'fig4')