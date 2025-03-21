import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Set proxy settings
os.environ['HTTP_PROXY'] = 'http://proxy-dmz.intel.com:912/'
os.environ['HTTPS_PROXY'] = 'http://proxy-dmz.intel.com:912/'

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis").to(torch_device)

model_inputs = tokenizer("World is so boring", return_tensors="pt").to(torch_device)
output = model(**model_inputs).logits.argmax(axis=1)

# Decode the output to get the sentiment label
labels = ["negative", "neutral", "positive"]
sentiment = labels[output.item()]

print(sentiment)