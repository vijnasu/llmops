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
python fine_tune_sentiment.py
```

## Files

1. **own_llm**
    - **Description**: This file contains code related to creating or managing your own language model (LLM).
    - **Purpose**: To develop and manage a custom language model.
    - **Key Functions**: Model creation, training, and evaluation functions specific to your custom LLM.

2. **sentiment_analysis**
    - **Description**: This file is focused on sentiment analysis tasks.
    - **Purpose**: To analyze the sentiment of given text data.
    - **Key Functions**: Functions to preprocess text, load sentiment analysis models, and predict sentiment.

3. **fine_tune_sentiment**
    - **Description**: This file contains the code to fine-tune a sentiment analysis model using the SST-2 dataset.
    - **Purpose**: To fine-tune a pre-trained sentiment analysis model and provide a function to analyze sentiment.
    - **Key Functions**:
        - `preprocess_function`: Preprocesses the dataset.
        - `analyze_sentiment`: Analyzes the sentiment of a given text.
        - Training and evaluation functions using the Hugging Face `Trainer` class.

4. **llm_compress**
    - **Description**: This file contains code related to compressing language models.
    - **Purpose**: To reduce the size of language models while maintaining their performance.
    - **Key Functions**: Functions for model compression techniques such as pruning, quantization, and distillation.