import re
import unicodedata
import pandas as pd
import os


def clean_text(text):
    """
    Clean up text:
        - Unicode standardization (NFC)
        - Move to normal
        - Remove URL
        - Remove special characters not in the Vietnamese alphabet
        - Remove spaces
    """
    text = str(text).lower()
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^0-9a-zA-ZÀ-Ỵà-ỵĐđ\s]", ' ', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text, stopwords):
    """
    Remove stop words from the text, but keep the emotional/negative words removed from the stopwords.
    """
    words = text.split()
    filtered = [w for w in words if w not in stopwords]
    return " ".join(filtered)


def preprocess_datasets(train_path, test_path, dev_path, stopwords_path, save_dir):
    """
    Apply cleaning pipeline to 3 train/test/dev sets.
    """
    # Load data
    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    test_df  = pd.read_csv(test_path, encoding="utf-8-sig")
    dev_df   = pd.read_csv(dev_path, encoding="utf-8-sig")

    # Load stopwords
    with open(stopwords_path, "r", encoding="utf-8") as f:
        stopwords = set(f.read().splitlines())

    # Remove some emotional and negative words from stopwords
    emotion_words = [
        # Negation, degree
        "không", "chẳng", "chả", "chưa", "đầy", "đủ", "rất", "hơi", "khá", "cực", "quá",
        "vô cùng", "hết sức", "cực kỳ", "siêu", "vô đối", "tương đối",
        # Positive
        "tốt", "đẹp", "hay", "tuyệt", "xuất sắc", "đỉnh", "ổn", "vui", "ưng", "dễ thương", "đáng yêu",
        "mượt", "chất lượng", "ấn tượng", "hiệu quả", "thích", "hấp dẫn", "hoàn hảo", "đáng tin",
        "đáng giá", "chuẩn", "xịn", "tuyệt vời", "ổn áp", "ngon", "ngon lành", "ok", "oke",
        # Negative
        "tệ", "xấu", "kém", "chán", "ghét", "bực", "dở", "tồi", "khó chịu", "kinh khủng",
        "thất vọng", "chậm", "lỗi", "rác", "ngu", "bực mình", "quá đáng", "bực tức", "dở ẹc",
        "chán đời", "kệch cỡm", "vô dụng", "phiền", "tồi tệ", "dở dang"
    ]
    for w in emotion_words:
        stopwords.discard(w)

    # Apply pipeline processing
    for df in [train_df, test_df, dev_df]:
        df["sentence"] = df["sentence"].apply(clean_text)
        df["sentence"] = df["sentence"].apply(lambda x: remove_stopwords(x, stopwords))

    # Create a folder to save the results
    os.makedirs(save_dir, exist_ok=True)

    # Save the result file
    train_df.to_csv(os.path.join(save_dir, "train_clean.csv"), index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(save_dir, "test_clean.csv"), index=False, encoding="utf-8-sig")
    dev_df.to_csv(os.path.join(save_dir, "dev_clean.csv"), index=False, encoding="utf-8-sig")

    print("✅ Đã xử lý và lưu dữ liệu vào:", save_dir)


if __name__ == "__main__":
    base_dir = "D:/Projects/Sentiment_Analysis/data"
    train_path = os.path.join(base_dir, "interim/train.csv")
    test_path = os.path.join(base_dir, "interim/test.csv")
    dev_path = os.path.join(base_dir, "interim/dev.csv")
    stopwords_path = os.path.join(base_dir, "external/vietnamese-stopwords.txt")
    save_dir = os.path.join(base_dir, "processed")

    preprocess_datasets(train_path, test_path, dev_path, stopwords_path, save_dir)
