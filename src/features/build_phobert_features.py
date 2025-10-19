import os
import pandas as pd
import torch
import numpy as np
import joblib
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# =============================
# CẤU HÌNH ĐƯỜNG DẪN
# =============================
DATA_DIR = "data/interim"
FEATURE_DIR = "data/features/PhoBERT"
LABEL_DIR = "models/PhoBERT"

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# =============================
# ĐỌC DỮ LIỆU
# =============================
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
dev_df   = pd.read_csv(os.path.join(DATA_DIR, "dev.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print("Train:", train_df.shape)
print("Dev:", dev_df.shape)
print("Test:", test_df.shape)

# =============================
# LOAD TOKENIZER VÀ MODEL
# =============================
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Nếu có GPU thì sử dụng
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# =============================
# HÀM LẤY EMBEDDING
# =============================
def get_phobert_embedding(texts, batch_size=8, max_len=256):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating PhoBERT embeddings"):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(device)

            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)

    return np.vstack(embeddings)


# =============================
# TẠO EMBEDDING CHO CÁC TẬP
# =============================
print("Bắt đầu tạo PhoBERT embeddings...")

X_train_pho = get_phobert_embedding(train_df["sentence"].tolist())
X_dev_pho   = get_phobert_embedding(dev_df["sentence"].tolist())
X_test_pho  = get_phobert_embedding(test_df["sentence"].tolist())

# =============================
# LƯU FILE RA DISK
# =============================
np.savez_compressed(os.path.join(FEATURE_DIR, "X_train_phobert.npz"), X_train_pho)
np.savez_compressed(os.path.join(FEATURE_DIR, "X_dev_phobert.npz"), X_dev_pho)
np.savez_compressed(os.path.join(FEATURE_DIR, "X_test_phobert.npz"), X_test_pho)

joblib.dump(train_df["sentiment"].values, os.path.join(LABEL_DIR, "y_train.pkl"))
joblib.dump(dev_df["sentiment"].values, os.path.join(LABEL_DIR, "y_dev.pkl"))
joblib.dump(test_df["sentiment"].values, os.path.join(LABEL_DIR, "y_test.pkl"))

print("✅ PhoBERT embeddings đã được lưu thành công!")
