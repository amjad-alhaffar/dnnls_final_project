# text-image model used to generate last frame using the desctiption

# the pretrained model for the text autoencoder using LSTM   
# @title The text autoencoder (Seq2Seq)

import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from utils import parse_gdi_text
from torch.utils.data import DataLoader
# @title Loading the dataset
train_dataset = load_dataset("daniel3303/StoryReasoning", split="train")
test_dataset = load_dataset("daniel3303/StoryReasoning", split="test")
# @title text-image dataset
from transformers import CLIPTokenizer
import re
clip_tokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-large-patch14"
)
def clean_dialogue(text):
    # remove anything inside quotes (dialogue)
    text = re.sub(r'".*?"', '', text)
    text = re.sub(r"'.*?'", '', text)

    # remove leftover punctuation noise
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ').strip()

    return text
class DiffusionFrameDatasetExport(Dataset):
    def __init__(self, raw_dataset, transforms=None, max_prompt_len=60):
        self.dataset = raw_dataset
        self.max_prompt_len = max_prompt_len

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

        self.samples = []
        for example in self.dataset:
            attrs = parse_gdi_text(example["story"])
            for i, img in enumerate(example["images"]):
                self.samples.append((img, attrs[i]["description"]))

    def __len__(self):
        return len(self.samples)

    def clean_prompt(self, text):
        text = text.lower()
        text = clean_dialogue(text)
        return " ".join(text.split()[:self.max_prompt_len])

    def __getitem__(self, idx):
        image, description = self.samples[idx]
        image = FT.equalize(image)
        image = self.transform(image)
        prompt = self.clean_prompt(description)

        return {
            "pixel_values": image,
            "prompt": prompt
        }
    
# @title For the text-image task

small_train_stories = train_dataset.select(range(200))
small_val_stories   = train_dataset.select(range(201, 240))


train_export_dataset = DiffusionFrameDatasetExport(
    small_train_stories,
    max_prompt_len=60
)

val_export_dataset = DiffusionFrameDatasetExport(
    small_val_stories,
    max_prompt_len=60
)
print("Train frames:", len(train_export_dataset))
print("Val frames:", len(val_export_dataset))
# Add data files to collab

import os, gc
os.makedirs("/content/data/train", exist_ok=True)
os.makedirs("/content/data/val", exist_ok=True)
import torchvision.transforms.functional as TF


def export_diffusion_dataset(dataset, out_dir, max_samples=1000):
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    N = len(dataset)

    for i in range(min(N, max_samples)):
        sample = dataset[i]   # ← INDEX ACCESS (important)

        image = sample["pixel_values"]
        prompt = sample["prompt"]

        pil_img = TF.to_pil_image(image)
        pil_img.save(os.path.join(out_dir, f"{saved:06d}.jpg"))

        with open(os.path.join(out_dir, f"{saved:06d}.txt"), "w") as f:
            f.write(prompt)

        # HARD cleanup
        del image, prompt, pil_img, sample
        gc.collect()

        saved += 1
        if saved % 100 == 0:
            print(f"Saved {saved}")

    print(f"Finished. Total saved: {saved}")

# @title Create json dataset
# this model expect using a metdadata json files also, so I need to generate it from my data
import json

base_dir = "/content/data/train"

metadata = []

for fname in sorted(os.listdir(base_dir)):
    if not fname.endswith(".jpg"):
        continue

    base = fname.replace(".jpg", "")
    txt_path = os.path.join(base_dir, base + ".txt")

    if not os.path.exists(txt_path):
        continue

    with open(txt_path, "r") as f:
        caption = f.read().strip()

    metadata.append({
        "file_name": fname,   # ✅ flat folder
        "text": caption
    })

with open(os.path.join(base_dir, "metadata.jsonl"), "w") as f:
    for item in metadata:
        f.write(json.dumps(item) + "\n")

print(f"Created metadata.jsonl with {len(metadata)} entries")

# @title Clone and install Diffusers
def trainingLoop():
    print("google colab only")
# !git clone https://github.com/huggingface/diffusers.git
# !pip uninstall -y diffusers
# !pip install git+https://github.com/huggingface/diffusers.git

# @title LoRA Text-image Training
# !accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora.py \
#   --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#   --dataset_name="imagefolder" \
#   --train_data_dir="/content/data/train" \
#   --caption_column="text" \
#   --resolution=256 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=8 \
#   --learning_rate=1e-4 \
#   --lr_scheduler="constant" \
#   --max_train_steps=500 \
#   --checkpointing_steps=500 \
#   --mixed_precision="fp16" \
#   --output_dir="/content/drive/MyDrive/lora-output"




if __name__ == "__main__":
    # @title Example text reconstruction task
    train_dataset = load_dataset("daniel3303/StoryReasoning", split="train")
    test_dataset = load_dataset("daniel3303/StoryReasoning", split="test")
    # only on colab 
    export_diffusion_dataset(
        train_export_dataset,
        "/content/data/train",
        max_samples=1000
    )

    export_diffusion_dataset(
        val_export_dataset,
        "/content/data/val",
        max_samples=200
    )
    trainingLoop()

