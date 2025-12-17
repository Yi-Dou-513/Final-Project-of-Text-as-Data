import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from datasets import Dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

os.chdir(r"E:\final")
print("Current working directory:", os.getcwd())

# ===========================
# 1. 读入数据
# ===========================
# 假设你从 R 导出了一个 csv: final_data_for_bert.csv
# 包含两列：text, label（取值: local/state/federal）
data_path = "final_data_for_bert.csv"
df = pd.read_csv(data_path)

# 丢掉缺失文本
df = df.dropna(subset=["text", "label"])

# 看一下标签分布（调试用，可以注释掉）
print(df["label"].value_counts())

# ===========================
# 2. train / validation / test 划分
# ===========================
# 这里简单用 70 / 15 / 15，可以自己改
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=123,
)

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=123,
)

print("Train size:", len(train_df))
print("Valid size:", len(valid_df))
print("Test size: ", len(test_df))

# ===========================
# 3. 标签编码：local/state/federal -> 0/1/2
# ===========================
unique_labels = sorted(df["label"].unique())  # 排个序保证稳定
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

print("Label mapping:", label2id)

for split_df in [train_df, valid_df, test_df]:
    split_df["label_id"] = split_df["label"].map(label2id)

# ===========================
# 4. 转成 HuggingFace Datasets
# ===========================
train_dataset = Dataset.from_pandas(
    train_df[["text", "label_id"]].rename(columns={"label_id": "labels"})
)
valid_dataset = Dataset.from_pandas(
    valid_df[["text", "label_id"]].rename(columns={"label_id": "labels"})
)
test_dataset = Dataset.from_pandas(
    test_df[["text", "label_id"]].rename(columns={"label_id": "labels"})
)

raw_datasets = DatasetDict(
    {
        "train": train_dataset,
        "validation": valid_dataset,
        "test": test_dataset,
    }
)

# ===========================
# 5. 加载 tokenizer 和 模型
# ===========================
# 你也可以换成 "bert-base-uncased"
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id,
)

# ===========================
# 6. 定义 tokenize 函数
# ===========================
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",  # 或者用 "longest"，但 max_length 更稳定
        max_length=128,        # 推文一般 128 足够，不够可以改大一点
    )

tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=["text"],  # 文本不用再留在模型输入里
)

# ===========================
# 7. 设置训练参数
# ===========================
batch_size = 16

training_args = TrainingArguments(
    output_dir="./bert_results",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
)

# ===========================
# 8. 定义评价函数（用 accuracy / macro F1）
# ===========================
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }

# ===========================
# 9. Trainer
# ===========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ===========================
# 10. 训练
# ===========================
trainer.train()

# ===========================
# 11. 在测试集上评估
# ===========================
pred_output = trainer.predict(tokenized_datasets["test"])
logits = pred_output.predictions
test_labels = pred_output.label_ids

test_preds = np.argmax(logits, axis=-1)

print("=== Classification report (test set) ===")
print(
    classification_report(
        test_labels,
        test_preds,
        target_names=[id2label[i] for i in range(len(unique_labels))],
        digits=3,
    )
)

print("=== Confusion matrix (test set) ===")
print(confusion_matrix(test_labels, test_preds))
