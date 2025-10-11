import pandas as pd
import os

def load_vsfc_split(split_path):

    # Read VSFC data from folder (train/dev/test), each folder contains 3 files: sents.txt, sentiments.txt, topics.txt
    sentiments_path = os.path.join(split_path, "sentiments.txt")
    sents_path = os.path.join(split_path, "sents.txt")
    topics_path = os.path.join(split_path, "topics.txt")

    # Read each file
    with open(sents_path, encoding="utf-8") as f:
        sents = f.read().splitlines()
    with open(sentiments_path, encoding="utf-8") as f:
        sentiments = f.read().splitlines()
    with open(topics_path, encoding="utf-8") as f:
        topics = f.read().splitlines()

    # Check line number
    assert len(sents) == len(sentiments) == len(topics), "⚠️ Số dòng không khớp giữa 3 file!"

    # Create DataFrame
    df = pd.DataFrame({
        "sentence": sents,
        "sentiment": sentiments,
        "topic": topics
    })
    return df


if __name__ == "__main__":
    # Train
    folder = "D:/Projects/Sentiment_Analysis/data/raw/train"
    df = load_vsfc_split(folder)
    os.makedirs("../../data/interim", exist_ok=True)
    df.to_csv("../../data/interim/train.csv", index=False)
    print(" Train data saved successfully!")

    # Test
    folder = "D:/Projects/Sentiment_Analysis/data/raw/test"
    df = load_vsfc_split(folder)
    df.to_csv("../../data/interim/test.csv", index=False)
    print(" Test data saved successfully!")

    # Dev
    folder = "D:/Projects/Sentiment_Analysis/data/raw/dev"
    df = load_vsfc_split(folder)
    df.to_csv("../../data/interim/dev.csv", index=False)
    print(" Dev data saved successfully!")
