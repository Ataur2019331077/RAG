# Tokenization Methods

Tokenization is a process of breaking text into smaller components such as words or subwords. Different tokenization algorithms exist, each with its own characteristics and use cases. Below are three commonly used methods:

## WordPiece
WordPiece is a subword tokenization algorithm that was originally developed for speech recognition and later adopted by models like BERT. It iteratively merges the most frequent character sequences into tokens based on their probability in a corpus.
- Used in models like BERT and RoBERTa.
- Reduces out-of-vocabulary (OOV) words.
- Balances between word-level and character-level representations.

## Unigram
Unigram tokenization is based on a probability model where each token is treated as an independent unit. It starts with a large vocabulary and iteratively removes tokens to optimize the likelihood of the training data.
- Used in models like LLaMA and DeepSeek.
- Allows for better flexibility with rare words.
- Uses **SentencePiece** for implementation.

## Byte-Pair Encoding (BPE)
BPE is a data compression technique adapted for tokenization. It starts with individual characters and merges the most frequent adjacent pairs until reaching the desired vocabulary size.
- Used in OpenAI's GPT models, including ChatGPT.
- Efficient in handling rare and compound words.
- Creates a balance between character-based and word-based tokenization.

## Tokenizer Comparison Table

| Model   | Tokenizer Type             | Underlying Method      |
|---------|----------------------------|------------------------|
| ChatGPT | `tiktoken`                 | BPE                    |
| LLaMA   | `PreTrainedTokenizerFast`  | Unigram (SentencePiece) |
| DeepSeek| `SentencePieceProcessor`   | Unigram (SentencePiece) |

## Tokenization Examples

### 1. Tokenization with ChatGPT (tiktoken, BPE)
```python
import tiktoken

# Load ChatGPT tokenizer
encoder = tiktoken.encoding_for_model("gpt-4")

# Tokenize sentence
sentence = "Tokenizing a sentence using tiktoken."
tokens = encoder.encode(sentence)
decoded_text = encoder.decode(tokens)

print("Tokenized IDs:", tokens)
print("Decoded Text:", decoded_text)
```

### 2. Tokenization with LLaMA (Hugging Face, Unigram)

```python
from transformers import AutoTokenizer

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Tokenize sentence
sentence = "Tokenizing a sentence using LLaMA."
token_ids = tokenizer.encode(sentence)
decoded_text = tokenizer.decode(token_ids)

print("Tokenized IDs:", token_ids)
print("Decoded Text:", decoded_text)
```

### 3. Tokenization with DeepSeek (SentencePiece, Unigram)

```python
import sentencepiece as spm

# Load DeepSeek tokenizer
sp = spm.SentencePieceProcessor()
sp.load("deepseek_tokenizer.model")

# Tokenize sentence
sentence = "Tokenizing a sentence using DeepSeek."
tokens = sp.encode(sentence, out_type=int)
decoded_text = sp.decode(tokens)

print("Tokenized IDs:", tokens)
print("Decoded Text:", decoded_text)
```

This document provides an overview of different tokenization methods and their implementations across popular language models.
