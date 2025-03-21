# Sentiment Analysis Fine-Tuning

This project fine-tunes a sentiment analysis model using the SST-2 dataset. The script `fine_tune_sentiment.py` trains the model and provides a function to analyze the sentiment of a given text.

## Requirements

- Python 3.6+
- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/sentiment-analysis-finetune.git
    cd sentiment-analysis-finetune
    ```

2. Install the required packages:
    ```sh
    pip install torch transformers datasets
    ```

## Usage

1. Train the model:
    ```sh
    python fine_tune_sentiment.py
    ```

2. Analyze sentiment:
    ```python
    from fine_tune_sentiment import analyze_sentiment

    sample_text = "The company reported a significant increase in profits."
    sentiment = analyze_sentiment(sample_text)
    print(f"Sentiment: {sentiment}")
    ```

## Script Details

### `fine_tune_sentiment.py`

- **Training the model:**
    ```python
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
    )

    trainer.train()
    ```

- **Evaluating the model:**
    ```python
    trainer.evaluate()
    ```

- **Analyzing sentiment:**
    ```python
    def analyze_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128).to(torch_device)
        outputs = model(**inputs).logits
        prediction = torch.argmax(outputs, dim=1).item()
        labels = ["negative", "positive"]
        return labels[prediction]
    ```

## Example

Run the script and analyze a sample text:
```sh
python 