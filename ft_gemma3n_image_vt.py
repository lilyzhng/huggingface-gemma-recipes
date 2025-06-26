import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
from functools import partial

import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Gemma3nForConditionalGeneration, Gemma3nProcessor


def collate_fn(examples, processor):
    messages = list()
    for sample in examples:
        image = sample["image"].convert("RGB")
        label = str(sample["label"])
        message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an assistant with great geometry skills.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": "How many intersection points are there in the image?",
                    },
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": label}]},
        ]
        messages.append(message)

    batch = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    # Mask the tokens that we do not want to include in the loss computation
    # -100 is ignored during categorical cross entropy loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == processor.tokenizer.image_token_id] = -100
    labels[labels == processor.tokenizer.boi_token_id] = -100
    labels[labels == processor.tokenizer.eoi_token_id] = -100

    batch["labels"] = labels

    return batch


def freeze_layers(model):
    for name, param in model.named_parameters():
        if "attn" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def run_inference(val_dataset, processor, model, fname):
    # infer before training
    val_sample = random.choice(val_dataset)
    image = val_sample["image"].convert("RGB")
    message = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an assistant with great geometry skills.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": "How many intersection points are there in the image?",
                },
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        generation = model.generate(**inputs, max_new_tokens=10, disable_compile=True)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)

    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Pred: {decoded}")
    plt.show()
    plt.savefig(f"outputs_fine_tune/{fname}")


def main():
    model_id = "gg-hf-gm/gemma-3n-E2B-it"
    processor = AutoProcessor.from_pretrained(model_id)

    # load the dataset
    dataset_id = "ariG23498/intersection-dataset"
    train_dataset = load_dataset(dataset_id, split="train")
    val_dataset = load_dataset(dataset_id, split="validation")

    # create data loader
    partial_collate_fn = partial(collate_fn, processor=processor)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        collate_fn=partial_collate_fn,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        collate_fn=partial_collate_fn,
    )

    # load the model and optimizer
    model = Gemma3nForConditionalGeneration.from_pretrained(model_id).to("cuda")

    run_inference(val_dataset, processor, model, "pred_before.png")

    model = freeze_layers(model)

    params_to_train = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params_to_train, lr=1e-5)

    # Start Training
    accumulation_steps = 8
    for idx, batch in tqdm(enumerate(train_dataloader)):
        outputs = model(**batch.to(model.device))
        loss = outputs.loss / accumulation_steps
        if idx % 50 == 0:
            val_loss = 0.0
            with torch.no_grad():
                count = 0
                for val_batch in val_dataloader:
                    val_loss = val_loss + model(**val_batch.to(model.device)).loss
                    count = count + 1
                val_loss = val_loss / count
            print(
                f"Iter: {idx} Loss: {loss.item():.4f} Val Loss: {val_loss.item():.4f}"
            )
            run_inference(val_dataset, processor, model, f"infer_{idx}.png")

        loss.backward()
        if idx % 8 == 0:
            optimizer.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    main()
