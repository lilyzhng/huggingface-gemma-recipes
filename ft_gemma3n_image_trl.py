"""
Train Gemma-3n on various vision-language datasets including intersection-dataset.

For Gemma-3n with intersection dataset:
accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    sft_vlm_gemma3n.py \
    --dataset_name ariG23498/intersection-dataset \
    --model_name_or_path gg-hf-gm/gemma-3n-E2B-it \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir gemma-3n-E2B-it-trl-sft-intersection \
    --bf16 \
    --torch_dtype bfloat16 \
    --use_peft \
    --lora_target_modules all-linear \
    --attn_implementation eager

Train Gemma-3n on the HuggingFaceH4/llava-instruct-mix-vsft dataset (single-image).

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    sft_vlm_gemma3n.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path google/gemma-3-4b-it \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir gemma-3-4b-it-trl-sft-llava-instruct-mix-vsft \
    --bf16 \
    --torch_dtype bfloat16 \
    --use_peft \
    --lora_target_modules all-linear \
    --attn_implementation eager

Train Gemma-3n on the FanqingM/MMIU-Benchmark dataset (multi-image).

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    sft_vlm_gemma3n.py \
    --dataset_name FanqingM/MMIU-Benchmark \
    --dataset_train_split test \
    --model_name_or_path google/gemma-3-4b-it \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir gemma-3-4b-it-trl-sft-MMIU-Benchmark \
    --bf16 \
    --torch_dtype bfloat16 \
    --use_peft \
    --lora_target_modules all-linear
    --attn_implementation eager
"""

import io
import os
import zipfile

import torch
from datasets import DatasetDict, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          Gemma3nForConditionalGeneration)
from trl import (ModelConfig, ScriptArguments, SFTConfig, SFTTrainer,
                 TrlParser, get_kbit_device_map, get_quantization_config)


def my_get_peft_config(model_args: ModelConfig):
    """A version of get_peft_config that handles comma-separated target modules"""
    if model_args.use_peft is False:
        return None

    # Import here to avoid issues if PEFT is not available
    try:
        from peft import LoraConfig
    except ImportError:
        raise ValueError(
            "You need to have PEFT library installed in your environment, make sure to install `peft`. "
            "Make sure to run `pip install -U peft`."
        )

    # Fix the target_modules to be a list if it's a comma-separated string
    target_modules = model_args.lora_target_modules
    if isinstance(target_modules, str) and target_modules != "all-linear":
        # Convert comma-separated string to list
        target_modules = [module.strip() for module in target_modules.split(",")]

    peft_config = LoraConfig(
        task_type=model_args.lora_task_type,
        r=model_args.lora_r,
        target_modules=target_modules,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        use_rslora=model_args.use_rslora,
        use_dora=model_args.use_dora,
        modules_to_save=model_args.lora_modules_to_save,
    )

    return peft_config


# For intersection dataset processing
def format_intersection_data(samples: dict) -> dict[str, list]:
    """Format intersection dataset to match expected message format"""
    formatted_samples = {"messages": []}
    for idx in range(len(samples["image"])):
        image = samples["image"][idx].convert("RGB")
        label = str(samples["label"][idx])

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
        formatted_samples["messages"].append(message)
    return formatted_samples


# For multi-image example
def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                if image is not None:
                    # Handle dictionary with bytes
                    if isinstance(image, dict) and "bytes" in image:
                        pil_image = Image.open(io.BytesIO(image["bytes"]))
                        image_inputs.append(pil_image.convert("RGB"))
                    # Handle PIL Image objects
                    elif hasattr(image, "convert"):
                        image_inputs.append(image.convert("RGB"))
    return image_inputs


def format_data(samples: dict) -> dict[str, list]:
    formatted_samples = {"messages": []}
    for cont in range(len(samples["question"])):
        images = []
        for img_path in samples["input_image_path"][cont]:
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append({"type": "image", "image": image})
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

        formatted_samples["messages"].append(
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": samples["context"][cont]}],
                },
                {
                    "role": "user",
                    "content": images
                    + [{"type": "text", "text": samples["question"][cont]}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": samples["output"][cont]}],
                },
            ]
        )
    return formatted_samples


# For multi-image example
def prepare_dataset(
    dataset: DatasetDict, dataset_name: str, dataset_train_split: str
) -> DatasetDict:
    all_files = list_repo_files(dataset_name, repo_type="dataset")
    zip_files = [f for f in all_files if f.endswith(".zip")]

    for zip_filename in zip_files:
        zip_path = hf_hub_download(
            repo_id=dataset_name, filename=zip_filename, repo_type="dataset"
        )
        extract_folder = zip_filename.replace(".zip", "")
        os.makedirs(extract_folder, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

    dataset = dataset.map(format_data, batched=True, batch_size=4, num_proc=16)
    return dataset


def main():
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    processor.tokenizer.padding_side = "right"

    # Use appropriate model class based on model name
    if "gemma-3n" in model_args.model_name_or_path.lower():
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )

    def collate_fn(examples):
        texts = []
        images_list = []

        for example in examples:
            # Apply chat template to get text
            text = processor.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            ).strip()
            texts.append(text)

            # Extract images
            if "images" in example:  # single-image case
                images = [img.convert("RGB") for img in example["images"]]
            else:  # multi-image case or intersection dataset
                images = process_vision_info(example["messages"])
            images_list.append(images)

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=images_list, return_tensors="pt", padding=True
        )

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()

        # Mask tokens for Gemma3n model
        if "gemma-3n" in model_args.model_name_or_path.lower():
            # Use Gemma3n specific token masking
            labels[labels == processor.tokenizer.pad_token_id] = -100
            if hasattr(processor.tokenizer, "image_token_id"):
                labels[labels == processor.tokenizer.image_token_id] = -100
            if hasattr(processor.tokenizer, "boi_token_id"):
                labels[labels == processor.tokenizer.boi_token_id] = -100
            if hasattr(processor.tokenizer, "eoi_token_id"):
                labels[labels == processor.tokenizer.eoi_token_id] = -100
        else:
            # Original masking for other models
            image_token_id = [
                processor.tokenizer.convert_tokens_to_ids(
                    processor.tokenizer.special_tokens_map["boi_token"]
                )
            ]
            labels[labels == processor.tokenizer.pad_token_id] = -100
            labels[labels == image_token_id] = -100
            labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Handle different dataset formats
    if script_args.dataset_name == "FanqingM/MMIU-Benchmark":
        dataset = prepare_dataset(
            dataset, script_args.dataset_name, script_args.dataset_train_split
        )
    elif script_args.dataset_name == "ariG23498/intersection-dataset":
        # Format intersection dataset
        dataset = dataset.map(
            format_intersection_data, batched=True, batch_size=4, num_proc=4
        )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None,
        processing_class=processor.tokenizer,
        peft_config=my_get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    main()
