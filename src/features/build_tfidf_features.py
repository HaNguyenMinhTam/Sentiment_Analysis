import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import scipy.sparse

def main():
    # --- 1️ Đường dẫn dữ liệu ---
    train_path = "D:/Projects/Sentiment_Analysis/data/processed/train_clean.csv"
    dev_path   = "D:/Projects/Sentiment_Analysis/data/processed/dev_clean.csv"
    test_path  = "D:/Projects/Sentiment_Analysis/data/processed/test_clean.csv"

    # --- 2️ Đọc dữ liệu ---
    train_df = pd.read_csv(train_path)
    dev_df   = pd.read_csv(dev_path)
    test_df  = pd.read_csv(test_path)

    # --- 3️ Chuẩn bị cột sentence ---
    X_train = train_df["sentence"].astype(str)
    X_dev   = dev_df["sentence"].astype(str)
    X_test  = test_df["sentence"].astype(str)

    # --- 4️ Tạo TF-IDF vectorizer ---
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    # --- 5️ Fit/transform ---
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_dev_tfidf   = vectorizer.transform(X_dev)
    X_test_tfidf  = vectorizer.transform(X_test)

    # --- 7️ Lưu vectorizer ---
    joblib.dump(vectorizer, "D:/Projects/Sentiment_Analysis/models/tfidf_vectorizer.pkl")

    # --- 8️ Lưu ma trận TF-IDF dạng nén ---
    scipy.sparse.save_npz("D:/Projects/Sentiment_Analysis/data/features/X_train_tfidf.npz", X_train_tfidf)
    scipy.sparse.save_npz("D:/Projects/Sentiment_Analysis/data/features/X_dev_tfidf.npz", X_dev_tfidf)
    scipy.sparse.save_npz("D:/Projects/Sentiment_Analysis/data/features/X_test_tfidf.npz", X_test_tfidf)

    print(" TF-IDF vectorizer và ma trận TF-IDF đã được lưu thành công!")

if __name__ == "__main__":
    main()