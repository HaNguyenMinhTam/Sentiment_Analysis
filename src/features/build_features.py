import re
import unicodedata
import pandas as pd
import os


def clean_text(text):
    """
    Làm sạch văn bản:
    - Chuyển về chữ thường
    - Xóa URL
    - Xóa ký tự đặc biệt
    - Xóa khoảng trắng thừa
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
                  r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]", '', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_vietnamese_accents(text):
    """
    Loại bỏ dấu tiếng Việt, chuyển 'đẹp quá' -> 'dep qua'
    """
    nfkd_form = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])


def remove_stopwords(text, stopwords):
    """
    Loại bỏ từ dừng khỏi văn bản.
    """
    words = text.split()
    filtered = [w for w in words if w not in stopwords]
    return " ".join(filtered)


def preprocess_datasets(train_path, test_path, dev_path, stopwords_path, save_dir):
    """
    Áp dụng toàn bộ quá trình làm sạch, bỏ dấu và loại từ dừng cho 3 tập train/test/dev.
    """
    # Load data
    train_df = pd.read_csv("D:/Projects/Sentiment_Analysis/data/interim/train.csv")
    test_df  = pd.read_csv("D:/Projects/Sentiment_Analysis/data/interim/test.csv")
    dev_df   = pd.read_csv("D:/Projects/Sentiment_Analysis/data/interim/dev.csv")

    # Load stopwords
    with open(stopwords_path, "r", encoding="utf-8") as f:
        stopwords = set(f.read().splitlines())

    # Apply cleaning pipeline
    for df in [train_df, test_df, dev_df]:
        df["sentence"] = df["sentence"].apply(clean_text)
        df["sentence"] = df["sentence"].apply(remove_vietnamese_accents)
        df["sentence"] = df["sentence"].apply(lambda x: remove_stopwords(x, stopwords))

    os.makedirs(save_dir, exist_ok=True)

    # Save processed files
    train_df.to_csv(os.path.join(save_dir, "train_clean.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test_clean.csv"), index=False)
    dev_df.to_csv(os.path.join(save_dir, "dev_clean.csv"), index=False)

    print(" Đã xử lý và lưu dữ liệu vào:", save_dir)
if __name__ == "__main__":
    # Khai báo đường dẫn input/output
    base_dir = "D:/Projects/Sentiment_Analysis/data"
    train_path = os.path.join(base_dir, "interim/train.csv")
    test_path = os.path.join(base_dir, "interim/test.csv")
    dev_path = os.path.join(base_dir, "interim/dev.csv")
    stopwords_path = os.path.join(base_dir, "external/vietnamese-stopwords.txt")
    save_dir = os.path.join(base_dir, "processed")

    # Gọi pipeline
    preprocess_datasets(train_path, test_path, dev_path, stopwords_path, save_dir)