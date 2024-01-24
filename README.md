# Sentiment-analysis-on-social-media
Cloning into 'Sentiment-Analysis-Benchmark'...
remote: Enumerating objects: 290, done.
remote: Counting objects: 100% (206/206), done.
remote: Compressing objects: 100% (154/154), done.
remote: Total 290 (delta 130), reused 95 (delta 49), pack-reused 84
Receiving objects: 100% (290/290), 30.55 MiB | 17.48 MiB/s, done.
Resolving deltas: 100% (158/158), done.
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting tokenizers==0.12.1
  Downloading tokenizers-0.12.1-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 35.3 MB/s eta 0:00:00
Collecting transformers==4.26.0
  Downloading transformers-4.26.0-py3-none-any.whl (6.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.3/6.3 MB 104.2 MB/s eta 0:00:00
Collecting datasets==2.5.1
  Downloading datasets-2.5.1-py3-none-any.whl (431 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 431.2/431.2 kB 48.2 MB/s eta 0:00:00
Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers==4.26.0) (3.12.0)
Collecting huggingface-hub<1.0,>=0.11.0 (from transformers==4.26.0)
  Downloading huggingface_hub-0.14.1-py3-none-any.whl (224 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 224.5/224.5 kB 26.6 MB/s eta 0:00:00
Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers==4.26.0) (1.22.4)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers==4.26.0) (23.1)
Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers==4.26.0) (6.0)
Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers==4.26.0) (2022.10.31)
...
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets==2.5.1) (2022.7.1)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.1->pandas->datasets==2.5.1) (1.16.0)
Installing collected packages: tokenizers, xxhash, multidict, frozenlist, dill, async-timeout, yarl, responses, multiprocess, huggingface-hub, aiosignal, transformers, aiohttp, datasets
Successfully installed aiohttp-3.8.4 aiosignal-1.3.1 async-timeout-4.0.2 datasets-2.5.1 dill-0.3.5.1 frozenlist-1.3.3 huggingface-hub-0.14.1 multidict-6.0.4 multiprocess-0.70.13 responses-0.18.0 tokenizers-0.12.1 transformers-4.26.0 xxhash-3.2.0 yarl-1.9.2
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
[{'text': 'no flights out of nashville today are you kidding me why are other airlines flying and you re not so frustrated',
  'label': 0},
 {'text': 'am but it says yall are sold out me amp my coworkers would need to get out first available',
  'label': 1},
 {'text': 'trying to change family vacation due to measles outbreak and haven been able to get anyone on the phone any help',
  'label': 0},
 {'text': 'zz', 'label': 1},
 {'text': 'you ve got mess here at dtw but your staff is doing great',
  'label': 2}]

 2) Import libraries
  from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification
import numpy as np
from transformers import AutoTokenizer
import torch
from datasets import Dataset
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import time
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import pandas as pd
import os
import re

3)Load dataset
tweet_train_path = "data/combine/kfolds_0/train.csv"
tweet_test_path = "data/combine/kfolds_0/test.csv"

train_df = pd.read_csv(tweet_train_path, nrows=9999999)
test_df = pd.read_csv(tweet_test_path, nrows=9999999)

train_df.head()

4) Data preprocessing
def emoji(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :') , :O
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\)|:O)', ' positiveemoji ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' positiveemoji ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' positiveemoji ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-; , @-)
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;|@-\))', ' positiveemoji ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:, :-/ , :-|
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:|:-/|:-\|)', ' negetiveemoji ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' negetiveemoji ', tweet)
    return tweet

def process_tweet(tweet):
    tweet = tweet.lower()                                            
    tweet = re.sub('@[^\s]+', '', tweet)                             
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', tweet)  
    tweet = re.sub(r"\d+", " ", str(tweet))                          
    tweet = re.sub('&quot;'," ", tweet)                             
    tweet = emoji(tweet)                                            
    tweet = re.sub(r"\b[a-zA-Z]\b", "", str(tweet))                 
    tweet = re.sub(r"[^\w\s]", " ", str(tweet))                     
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet)                       
    tweet = re.sub(r"\s+", " ", str(tweet)) .strip()
    return tweet
train_df["text"] = train_df["text"].apply(lambda x: process_tweet(x))
test_df["text"] = test_df["text"].apply(lambda x: process_tweet(x))
train_df.head()
# Convert data to trainable format
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {v: k for k, v in id2label.items()}

train_df["label"] = train_df["label"].apply(lambda x: label2id[x])
test_df["label"] = test_df["label"].apply(lambda x: label2id[x])

train_texts = train_df["text"].values
train_labels = train_df["label"].values

test_texts = test_df["text"].values
test_labels = test_df["label"]. values

train_data = [
    {"text": text, "label": label}
    for text, label in zip(train_texts, train_labels)
]

test_data = [
    {"text": text, "label": label}
    for text, label in zip(test_texts, test_labels)
]

train_data[:5]

5) ### Draw WordCloud
# from wordcloud import WordCloud

# positive_words = " ".join([x['text'] for x in train_data if x['label'] == label2id['positive']])
# wordcloud = WordCloud(
#     width=800,
#     height=500,
#     random_state=21,
#     max_font_size=110,
#     background_color="rgba(255, 255, 255, 0)",
#     mode="RGBA",
# ).generate(positive_words)
# plt.figure(dpi=100)
# plt.figure(figsize=(5, 4))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.title("Most Used Positive Words")
# plt.show()
# negative_words = " ".join([x['text'] for x in train_data if x['label'] == label2id['negative']])
# wordcloud = WordCloud(
#     width=800,
#     height=500,
#     random_state=21,
#     max_font_size=110,
#     background_color="rgba(255, 255, 255, 0)",
#     mode="RGBA",
# ).generate(negative_words)
# plt.figure(dpi=100)
# plt.figure(figsize=(5, 4))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.title("Most Used Negative Words")
# plt.show()
# neutral_words = " ".join([x['text'] for x in train_data if x['label'] == label2id['neutral']])
# wordcloud = WordCloud(
#     width=800,
#     height=500,
#     random_state=21,
#     max_font_size=110,
  background_color="rgba(255, 255, 255, 0)",
 mode="RGBA",
).generate(neutral_words)
plt.figure(dpi=100)
plt.figure(figsize=(5, 4))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Used Neutral Words")
plt.show()

6)  Training model
#### Load pretrained model
model_name = "roberta-base"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "negative", 1: "neutral", 2: "positive"}
#### Init trainer
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

train_data = Dataset.from_list(train_data)

tokenized_train_data = train_data.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
#### Start training
straining_time = time.time()
trainer.train()
training_time = time.time() - straining_time

print(f"Training time: {training_time:.3f}s")
# save model
output_model_path = f"models/twitter-{model_name}"
os.makedirs("models/", exist_ok=True)
trainer.save_model(output_model_path)

print(f"Model has been saved to {output_model_path}")
#### Evaluate
def evaluate(test_data):
      test_data = Dataset.from_list(test_data)
      tokenized_test_data = test_data.map(preprocess_function, batched=True)

      training_args = TrainingArguments(
          output_dir="./results",
          per_device_eval_batch_size=16,
          num_train_epochs=0
      )

      trainer = Trainer(
          model=model,
          args=training_args,
          tokenizer=tokenizer,
          data_collator=data_collator,
      )

      stime = time.time()
      result = trainer.predict(tokenized_test_data)
      inference_time = (time.time() - stime) / len(test_data)

      result = result.predictions.argmax(axis=1)

      y_test = [x["label"] for x in tokenized_test_data]

      acc = accuracy_score(y_test, result)

      p_micro = precision_score(y_test, result, average="micro")
      p_macro = precision_score(y_test, result, average="macro")

      r_micro = recall_score(y_test, result, average="micro")
      r_macro = recall_score(y_test, result, average="macro")

      f1_micro = f1_score(y_test, result, average="micro")
      f1_macro = f1_score(y_test, result, average="macro")

      return {
            "acc": acc,
            "p_micro": p_micro,
            "p_macro": p_macro,
            "r_micro": r_micro,
            "r_macro": r_macro,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "inference_time": inference_time,
            "y_predict": result,
            "y_true": y_test
        }


metrics = evaluate(test_data)
print()
print(f"Accuracy: {metrics['acc']:.3f}")
print(f"Precision-micro: {metrics['p_micro']:.3f} Precision-macro: {metrics['p_macro']:.3f}")
print(f"Recall-micro: {metrics['r_micro']:.3f} Recall-macro: {metrics['r_macro']:.3f}")
print(f"F1-micro: {metrics['f1_micro']:.3f} F1-macro: {metrics['f1_macro']:.3f}")
print(f"Execution time: {metrics['inference_time']*1000:.3f} ms/sample")

7) Draw pie chart
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly import tools

def compute_label_ratio(y):
  value, count = np.unique(y, return_counts=True)
  ratio = count / count.sum()
  return value, ratio


value, labelled_ratio = compute_label_ratio(metrics['y_true'])
_, predict_ratio = compute_label_ratio(metrics['y_predict'])
x = [id2label[id] for id in value]

colors = ['rgb(239,85,59)', 'rgb(99,110,250)', 'rgb(0,204,150)']
fig = tools.make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=x, values=labelled_ratio, name="Labelled Data", marker={'colors': colors}),
          1, 1)
fig.add_trace(go.Pie(labels=x, values=predict_ratio, name="Predicted Data", marker={'colors': colors}),
          1, 2)

fig.update_traces(hole=.6, hoverinfo="label+percent+name")
fig.update_layout(
    autosize=False,
    width=800,
    height=500,
    title=dict(
        text="Sentiment distribution by type in Twitter dataset",
        x=0.22,
        y=0.82
    ),
    legend=dict(
        font=dict(size=16),
        yanchor="middle",
        xanchor="right",
        x=1.22,
        y=0.5
    ),
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Labelled Data', x=0.093, y=0.5, font_size=16, showarrow=False),
                 dict(text='Predicted Data', x=0.893, y=0.5, font_size=16, showarrow=False)])

fig.show()

