import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import os

# Load data 
train_df = pd.read_csv("D:/Projects/Sentiment_Analysis/data/processed/train_clean.csv")
dev_df = pd.read_csv("D:/Projects/Sentiment_Analysis/data/processed/dev_clean.csv")
test_df = pd.read_csv("D:/Projects/Sentiment_Analysis/data/processed/test_clean.csv")

print("Train:", train_df.shape)
print("Dev:", dev_df.shape)
print("Test:", test_df.shape)

# Ensure data type and word separation
train_df['sentence'] = train_df['sentence'].astype(str)
dev_df['sentence'] = dev_df['sentence'].astype(str)
test_df['sentence'] = test_df['sentence'].astype(str)

sentences = [text.split() for text in train_df['sentence']]
dev_sentences = [text.split() for text in dev_df['sentence']]
test_sentences = [text.split() for text in test_df['sentence']]

# Fit the Word2Vec model
w2v_model = Word2Vec(
    sentences=sentences,   # training data
    vector_size=100,       # vector dimensions (usually 100–300)
    window=5,              # ngữ cảnh (context size)
    min_count=2,           # ignore words that appear less than 2 times
    workers=4,             # number of CPU threads
    sg=1                   # skip-gram (1) or CBOW (0)
)

# Function to get the average vector of a sentence
def get_sentence_vector(tokens, model):
    valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(valid_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(valid_vectors, axis=0)

# Represent the entire sentence as a vector
X_train_w2v = np.array([get_sentence_vector(tokens, w2v_model) for tokens in sentences])
X_dev_w2v = np.array([get_sentence_vector(tokens, w2v_model) for tokens in dev_sentences])
X_test_w2v = np.array([get_sentence_vector(tokens, w2v_model) for tokens in test_sentences])

# Save model and feature vector
w2v_model.save("D:/Projects/Sentiment_Analysis/models/word2vec.model")
print("✅ Word2Vec model saved as 'word2vec.model'")
joblib.dump(train_df["sentiment"].values, "D:/Projects/Sentiment_Analysis/models/Word2Vec/y_train.pkl")
joblib.dump(dev_df["sentiment"].values, "D:/Projects/Sentiment_Analysis/models/Word2Vec/y_dev.pkl")
joblib.dump(test_df["sentiment"].values, "D:/Projects/Sentiment_Analysis/models/Word2Vec/y_test.pkl")

# Lưu ma trận TF-IDF dạng nén (nếu muốn)
np.savez_compressed("D:/Projects/Sentiment_Analysis/data/features/Word2Vec/X_train_w2v.npz", X_train_w2v)
np.savez_compressed("D:/Projects/Sentiment_Analysis/data/features/Word2Vec/X_dev_w2v.npz", X_dev_w2v)
np.savez_compressed("D:/Projects/Sentiment_Analysis/data/features/Word2Vec/X_test_w2v.npz", X_test_w2v)

print("✅ Saved vectorized data: X_train_w2v.npy, X_dev_w2v.npy, X_test_w2v.npy")
