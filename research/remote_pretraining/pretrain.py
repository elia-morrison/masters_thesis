# MLM pretraining

from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import collections
import numpy as np
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer, pipeline
from transformers.data.data_collator import default_data_collator

model_checkpoint = "roberta-large"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

mlm_data_train = "mlm_data_train.jsonl"
mlm_data_val = "mlm_data_val.jsonl"

TRAIN_LEN = 22236629
VAL_LEN = 2000
# found experimentally. 
# all datapoints are contatenated during training, 
# therefore each sequence of len 512 is actually more than one datapoint
AVG_DATAPOINT_LEN = 64
ACTUAL_TRAIN_LEN = TRAIN_LEN // (512 // AVG_DATAPOINT_LEN)
ACTUAL_VAL_LEN = VAL_LEN // (512 // AVG_DATAPOINT_LEN)

BATCH_SIZE = 1
NUM_STEPS_PER_EPOCH = ACTUAL_TRAIN_LEN // BATCH_SIZE
EVALUATION_STEPS = (ACTUAL_VAL_LEN * 19) // BATCH_SIZE
OUTPUT_DIR = './output'

print(f"EVAL_STEPS = {EVALUATION_STEPS}")

mlm_dataset = load_dataset("json", data_files={
    'train': mlm_data_train, 
    'validation': mlm_data_val}, 
    streaming=True)

def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

tokenized_datasets = mlm_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

chunk_size = 512

def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
wwm_probability = 0.2

def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, 
    evaluation_strategy="steps",
    save_steps=EVALUATION_STEPS,
    max_steps=NUM_STEPS_PER_EPOCH,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=EVALUATION_STEPS,
    report_to="tensorboard")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()