import os
import re
import unicodedata
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import joblib
from transformers import AutoTokenizer, AutoModel


# =============================
# ⚙️ CẤU HÌNH ĐƯỜNG DẪN
# =============================
BASE_DIR = "D:/Projects/Sentiment_Analysis"
DATA_DIR = os.path.join(BASE_DIR, "data/interim")        # đọc dữ liệu gốc (chưa TF-IDF clean)
FEATURE_DIR = os.path.join(BASE_DIR, "data/features/PhoBERT")
LABEL_DIR = os.path.join(BASE_DIR, "models/PhoBERT")

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)


# =============================
# 🧹 HÀM TIỀN XỬ LÝ VĂN BẢN
# =============================
def preprocess_phobert_text(text):
    """
    Chuẩn hóa văn bản trước khi đưa vào PhoBERT:
    - Chuẩn Unicode (NFC)
    - Chuyển về chữ thường
    - Xóa ký tự đặc biệt, giữ lại chữ, số và dấu tiếng Việt
    - Chuẩn hóa khoảng trắng
    """
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.lower()
    text = re.sub(r'[^0-9a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệ'
                  r'ìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữự'
                  r'ỳýỷỹỵđ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =============================
# 📖 ĐỌC DỮ LIỆU
# =============================
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
dev_df   = pd.read_csv(os.path.join(DATA_DIR, "dev.csv"), encoding="utf-8-sig")
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig")

print(f"Train: {train_df.shape}, Dev: {dev_df.shape}, Test: {test_df.shape}")

# Làm sạch cột 'sentence'
for df in [train_df, dev_df, test_df]:
    df["sentence_clean"] = df["sentence"].apply(preprocess_phobert_text)


# =============================
# 🤖 LOAD PHOBERT MODEL
# =============================
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Using device: {device}")


# =============================
# 🔢 HÀM SINH EMBEDDING
# =============================
def get_phobert_embedding(texts, batch_size=8, max_length=256, pooling="cls"):
    """
    Tạo embedding PhoBERT từ danh sách câu.
    - pooling: 'cls' hoặc 'mean'
    """
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating PhoBERT embeddings"):
            batch_texts = texts[i:i+batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            outputs = model(**encoded)
            hidden = outputs.last_hidden_state

            if pooling == "cls":
                batch_embed = hidden[:, 0, :].cpu().numpy()
            elif pooling == "mean":
                batch_embed = hidden.mean(dim=1).cpu().numpy()
            else:
                raise ValueError("pooling phải là 'cls' hoặc 'mean'")

            embeddings.append(batch_embed)

    return np.vstack(embeddings)


# =============================
# 🚀 TẠO EMBEDDING CHO 3 TẬP
# =============================
print("Bắt đầu trích xuất embedding PhoBERT...")

X_train_pho = get_phobert_embedding(train_df["sentence_clean"].tolist(), batch_size=8, max_length=256)
X_dev_pho   = get_phobert_embedding(dev_df["sentence_clean"].tolist(), batch_size=8, max_length=256)
X_test_pho  = get_phobert_embedding(test_df["sentence_clean"].tolist(), batch_size=8, max_length=256)

print("✅ Đã sinh xong embedding PhoBERT!")


# =============================
# 💾 LƯU EMBEDDING & LABEL
# =============================
np.savez_compressed(os.path.join(FEATURE_DIR, "X_train_phobert.npz"), X_train_pho)
np.savez_compressed(os.path.join(FEATURE_DIR, "X_dev_phobert.npz"), X_dev_pho)
np.savez_compressed(os.path.join(FEATURE_DIR, "X_test_phobert.npz"), X_test_pho)

joblib.dump(train_df["sentiment"].values, os.path.join(LABEL_DIR, "y_train.pkl"))
joblib.dump(dev_df["sentiment"].values, os.path.join(LABEL_DIR, "y_dev.pkl"))
joblib.dump(test_df["sentiment"].values, os.path.join(LABEL_DIR, "y_test.pkl"))

print("🎯 Embedding & labels đã được lưu tại:", FEATURE_DIR)
