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
from transformers import TrainerCallback, TrainerState, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import log_loss
from datasets import load_dataset
import torch
import numpy as np
import wandb
import os


model_str = "microsoft/phi-2"
# model_str = "microsoft/Phi-3-mini-4k-instruct"
# model_str = "facebook/galactica-125m"
run_number = 1

run_name = f"{model_str.replace('/','-')}{run_number}"

os.environ["WANDB_PROJECT"] = "llm-human-preference"

model = AutoModelForSequenceClassification.from_pretrained(
    model_str,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    num_labels=3,
)
tokenizer = AutoTokenizer.from_pretrained(model_str)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

model.resize_token_embeddings(len(tokenizer))

tokenizer.pad_token = (
    tokenizer.pad_token if tokenizer.pad_token else tokenizer.eos_token
)
model.config.pad_token_id = tokenizer.pad_token_id

dataset = load_dataset("lmsys/lmsys-arena-human-preference-55k")

max_length = 1000
batch_size = 12

# split dataset
dataset = dataset["train"]
# use small subset for testing
dataset = dataset.select(range(16000))
dataset = dataset.train_test_split(test_size=0.1)


def apply_template(prompts, responses_a, responses_b):
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
    return f"Rate who is better(1/2/tie)\n<Model 1>\n{conversation_1}\n\n<Model 2>\n{conversation_2}\n\nThe winner is"


def preprocess_function(examples):
    model_inputs = []
    labels = []
    for example in zip(
        examples["prompt"],
        examples["response_a"],
        examples["response_b"],
        examples["winner_model_a"],
        examples["winner_model_b"],
    ):
        (
            prompts,
            responses_a,
            responses_b,
            winner_model_a,
            winner_model_b,
        ) = example
        model_input = apply_template(prompts, responses_a, responses_b)
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
    model_inputs = apply_template(prompts, responses_a, responses_b)
    tokens = tokenizer(model_inputs, padding=True, truncation=False)
    return len(tokens["input_ids"]) <= max_length


filtered_dataset = dataset.filter(filter_function)

# Preprocess the filtered dataset
dataset = filtered_dataset.map(preprocess_function, batched=True, batch_size=batch_size)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


class LogLossCallback(TrainerCallback):
    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control, metrics, **kwargs
    ):
        if state.is_local_process_zero:
            wandb.log({"val_loss": metrics["eval_loss"]})


def compute_log_loss(pred: EvalPrediction):
    logits, labels = pred
    # Ensure logits are probabilities
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
    # Compute log loss
    return {"eval_loss": log_loss(labels, probabilities)}


# Update TrainingArguments to use logging_steps and evaluation_strategy

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=0,
    weight_decay=0.002,
    logging_dir="./logs",
    logging_steps=5,
    report_to="wandb",
    run_name=run_name,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    eval_steps=20,  # evaluate every 50 steps
    save_steps=200000,  # save checkpoint every 50 steps
    save_total_limit=2,  # limit number of total saved checkpoints
    learning_rate=3e-5,
    max_grad_norm=5.0,
)

# Adding the callback to the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_log_loss,
    callbacks=[LogLossCallback],
)

trainer.train()
trainer.save_model("results")
