# Hugging Face Gemma Recipes

![repository thumbnail](../assets/thumbnail.png)

🤗💎 Welcome! This repository contains *minimal* recipes to get started quickly with the Gemma family of models.

> [!Note]
> Fine tune Gemma 3n on a Free Colab Notebook: <a href="https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/fine_tune_gemma3n_on_t4.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>


## Getting Started

The easiest way to quickly run a Gemma 💎 model on your machine would be with the
🤗 `transformers` repository. Make sure you have the latest release installed.

```shell
$ pip install -U -q transformers timm
```

Once we've installed the dependencies, we can use the model creating a common function `model_generation`:

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_id = "google/gemma-3n-e4b-it" # google/gemma-3n-e2b-it
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id).to(device)

def model_generation(model, messages):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_len = inputs["input_ids"].shape[-1]

    inputs = inputs.to(model.device, dtype=model.dtype)

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=32, disable_compile=False)
        generation = generation[:, input_len:]

    decoded = processor.batch_decode(generation, skip_special_tokens=True)
    print(decoded[0])
```

And then using calling it with our specific modality:

**Text only**

```python
# Text Only

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is the capital of France?"}
        ]
    }
]
model_generation(model, messages)
```

**Interleaved with Audio**

```python
# Interleaved with Audio

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the following speech segment in English:"},
            {"type": "audio", "audio": "https://huggingface.co/datasets/ariG23498/demo-data/resolve/main/speech.wav"},
        ]
    }
]
model_generation(model, messages)
```

**Interleaved with Image/Video**

```python
# Interleaved with Image

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/ariG23498/demo-data/resolve/main/airplane.jpg"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]
model_generation(model, messages)
```

## Model Fine Tuning

We include a series of notebook+scripts for fine tuning the models.

* [Fine tuning Gemma 3n on free Colab T4 (notebook)](/notebooks/fine_tune_gemma3n_on_t4.ipynb) <a href="https://colab.research.google.com/github/huggingface/huggingface-gemma-recipes/blob/main/notebooks/fine_tune_gemma3n_on_t4.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
* [Fine tuning Gemma 3n on audio (notebook)](/notebooks/fine_tune_gemma3n_on_audio.ipynb)
* [Fine tuning Gemma 3n on images](/scripts/ft_gemma3n_image_vt.py)
* [Fine tuning Gemma 3n on audio](/scripts/ft_gemma3n_audio_vt.py)
* [Fine tuning Gemma 3n on images using TRL](/scripts/ft_gemma3n_image_trl.py)

Before fine-tuning the model, ensure all dependencies are installed:

```bash
$ pip install -U -q -r requirements.txt
```

