# use phi-3 mini model
# add value head with two output values
# use log loss
# connect to wandb

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
)
from datasets import load_dataset
import torch
import numpy as np
import wandb
import os

# model_str = "microsoft/Phi-3-mini-4k-instruct"
model_str = "facebook/galactica-125m"
run_number = 1

os.environ["WANDB_PROJECT"] = "llm-human-preference"

model = AutoModelForSequenceClassification.from_pretrained(
    model_str,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    num_labels=3,
)
tokenizer = AutoTokenizer.from_pretrained(model_str)
tokenizer.add_special_tokens(
    {
        "pad_token": "<pad>",
        # "additional_special_tokens": [
        #     "<user>",
        #     "<model>",
        #     "<CONVERSATION 1>",
        #     "<CONVERSATION 2>",
        # ],
    }
)  # add pad token to tokenizer for padding
dataset = load_dataset("lmsys/lmsys-arena-human-preference-55k")

max_length = 1000
batch_size = 24

# split dataset
dataset = dataset["train"]
# use small subset for testing
dataset = dataset.select(range(1000))
dataset = dataset.train_test_split(test_size=0.1)


def preprocess_function(examples):
    model_inputs = []
    labels = []
    for example in zip(
        examples["prompt"],
        examples["response_a"],
        examples["response_b"],
        examples["winner_model_a"],
        examples["winner_model_b"],
        examples["winner_tie"],
    ):
        (
            prompts,
            responses_a,
            responses_b,
            winner_model_a,
            winner_model_b,
            winner_tie,
        ) = example
        conversation_1 = "".join(
            [
                f"<user>{prompt}<model>{response}"
                for prompt, response in zip(prompts, responses_a)
            ]
        )
        conversation_2 = "".join(
            [
                f"<user>{prompt}<model>{response}"
                for prompt, response in zip(prompts, responses_b)
            ]
        )
        model_input = (
            f"<CONVERSATION 1>\n{conversation_1}\n\n<CONVERSATION 2>\n{conversation_2}"
        )
        tokens = tokenizer(
            model_input, padding="max_length", truncation=False, max_length=max_length
        )
        model_inputs.append(tokens)
        label = 0 if winner_model_a else 1 if winner_model_b else 2
        labels.append(label)

    # Convert lists of dictionaries to a dictionary of lists
    input_ids = [x["input_ids"] for x in model_inputs]
    attention_mask = [x["attention_mask"] for x in model_inputs]

    # Ensure the output structure is correct
    batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return batch


# Filter the dataset
def filter_function(example):
    prompts = example["prompt"]
    responses_a = example["response_a"]
    responses_b = example["response_b"]
    conversation_1 = "".join(
        [
            f"<user>{prompt}<model>{response}"
            for prompt, response in zip(prompts, responses_a)
        ]
    )
    conversation_2 = "".join(
        [
            f"<user>{prompt}<model>{response}"
            for prompt, response in zip(prompts, responses_b)
        ]
    )
    model_inputs = (
        f"<CONVERSATION 1>\n{conversation_1}\n\n<CONVERSATION 2>\n{conversation_2}"
    )

    tokens = tokenizer(model_inputs, padding=True, truncation=False)
    return len(tokens["input_ids"]) <= max_length


filtered_dataset = dataset.filter(filter_function)

# Preprocess the filtered dataset
dataset = filtered_dataset.map(preprocess_function, batched=True, batch_size=batch_size)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="wandb",
    run_name=f"{model_str}{run_number}",
    gradient_accumulation_steps=1,
)


# use log loss
def compute_log_loss(eval_pred):
    logits, labels = eval_pred
    # Ensure logits are probabilities
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

    # Compute log loss
    log_loss = -np.sum(labels * np.log(probabilities)) / len(labels)
    return log_loss


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_log_loss,
)

trainer.train()
trainer.save_model("results")
