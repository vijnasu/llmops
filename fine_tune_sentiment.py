import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
from datasets import load_dataset

# Set proxy settings
os.environ['HTTP_PROXY'] = 'http://proxy-dmz.intel.com:912/'
os.environ['HTTPS_PROXY'] = 'http://proxy-dmz.intel.com:912/'

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the tokenizer and model for sequence classification
model_name = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis").to(torch_device)

# Load the dataset
dataset = load_dataset("sst2")

# Preprocess the data
def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
)

trainer.train()

# Evaluate the model
trainer.evaluate()

# Example usage
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128).to(torch_device)
    outputs = model(**inputs).logits
    prediction = torch.argmax(outputs, dim=1).item()
    labels = ["negative", "positive"]  # SST-2 dataset has only negative and positive labels
    return labels[prediction]

if __name__ == "__main__":
    sample_text = "The company reported a significant increase in profits."
    sentiment = analyze_sentiment(sample_text)
    print(f"Sentiment: {sentiment}")