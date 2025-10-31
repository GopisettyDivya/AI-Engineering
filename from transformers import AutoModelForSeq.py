from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# 1) Load base model
model_name = "google/flan-t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2) Prepare dataset (CSV with 'question','answer')
dataset = load_dataset("csv", data_files="company_faq.csv")

def preprocess(batch):
    inputs = [q for q in batch["question"]]
    targets = [a for a in batch["answer"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess, batched=True)

# 3) Training setup
training_args = TrainingArguments(
    output_dir="./sllm",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
)

# 4) Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["train"].select(range(50))  # quick eval
)
trainer.train()

# 5) Save
trainer.save_model("./company-sllm")
tokenizer.save_pretrained("./company-sllm")
