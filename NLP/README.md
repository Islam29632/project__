# NLP Projects

This repository contains three distinct Natural Language Processing (NLP) projects, each demonstrating different facets of NLP, from machine translation to named entity recognition and word embeddings.

## 1. English-French Neural Machine Translation

This project implements a sequence-to-sequence (Seq2Seq) model with an Encoder-Decoder architecture using LSTMs for translating English sentences to French.

### Dataset
The project utilizes a sample of 500,000 parallel sentences from the `europarl-v7.fr-en` dataset, available on Kaggle.

### Preprocessing and Tokenization
- **SentencePiece Tokenization**: Both English and French sentences are tokenized using SentencePiece, which handles subword tokenization, essential for out-of-vocabulary words. Special `<start>` and `<end>` tokens are added to the French (decoder) sequences.
- **Padding**: Sequences are padded to a maximum length of 30 for English and 32 (30 + 2 for special tokens) for French, ensuring uniform input dimensions for the neural network.
- **Decoder Input/Target Shift**: The decoder input is shifted to the right by one timestep to create the target sequence for training.

### Model Architecture
The translation model consists of:
- **Encoder**: An Embedding layer followed by an LSTM layer that processes the English input and outputs the hidden and cell states.
- **Decoder**: Another Embedding layer and an LSTM layer that takes the encoder's final states as its initial states. It then processes the French input sequence (shifted) and outputs sequences.
- **Dense Layer with Softmax**: The decoder's output is passed through a Dense layer with a softmax activation to predict the probability distribution over the French vocabulary.

### Training
The model is compiled with the Adam optimizer and `sparse_categorical_crossentropy` loss. It's trained for 10 epochs with a batch size of 32 and a validation split of 0.1, utilizing early stopping. An accuracy of approximately **0.4170** was achieved.

---

## 2. Named Entity Recognition (NER) and Part-of-Speech (POS) Tagging

This project demonstrates Named Entity Recognition (NER) and Part-of-Speech (POS) tagging using the spaCy library on a sample text.

### Preprocessing
The text is preprocessed (details not provided in the snippet, but assumed to be standard cleaning steps).

### SpaCy Model
The `en_core_web_md` spaCy model is loaded for performing NER and POS tagging.

### Entity and POS Tagging
- **Named Entity Recognition**: The code iterates through processed documents (`docs`) and prints recognized entities along with their labels (e.g., `PERSON`, `GPE`, `ORG`).
- **Part-of-Speech Tagging**: For each token in the documents (`tag_docs`), its text and corresponding POS tag (e.g., `NOUN`, `VERB`, `ADJ`) are printed.

### Visualization of POS Tag Frequencies
The project includes code to visualize the frequency distribution of POS tags for several sentences using bar plots, providing insights into the grammatical structure of the text.

---

## 3. Word2Vec for Book Review Analysis

This project explores the application of the Word2Vec model for generating word embeddings from a dataset of book reviews, allowing for semantic similarity analysis.

### Dataset
The project uses the `all_data.csv` file, presumably containing book reviews and ratings.

### Data Preprocessing and EDA
- **Cleaning and Preprocessing**: The reviews are cleaned and preprocessed (details not provided, but typically involve lowercasing, punctuation removal, stop word removal, and tokenization).
- **Rating Distribution**: A pie chart visualizes the distribution of the top 10 book ratings, offering an overview of user sentiment.
- **Word Cloud**: A word cloud is generated from the preprocessed `reviews_prep` data, highlighting the most frequent words in the dataset.

### Word2Vec Model
- **Model Training**: A Word2Vec model is trained on the preprocessed `reviews_prep` data with the following parameters:
  - `vector_size=100`: Each word is represented by a 100-dimensional vector.
  - `window=5`: The maximum distance between the current and predicted word within a sentence.
  - `min_count=3`: Ignores all words with a total frequency lower than this.
  - `workers=4`: Number of CPU cores to use for training.
- **Model Saving/Loading**: The trained model is saved as `text_review_word2vec.model` and can be loaded for later use.

### Word Embeddings and Similarity
- **Vector Representation**: The project demonstrates how to retrieve the vector representation for a specific word (e.g., `'great'`).
- **Similar Words**: It identifies and prints the top 3 most similar words to a given word (`'great'`) based on their vector proximity.
- **PCA for Visualization**: To visualize the semantic relationships, the top 20 words similar to `'great'` (including `'great'` itself) are selected, their vectors are retrieved, and then Principal Component Analysis (PCA) is used to reduce the dimensionality to 2D for easier plotting.
