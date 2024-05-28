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