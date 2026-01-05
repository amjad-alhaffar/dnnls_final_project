from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
from utils import parse_gdi_text
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
from torch.optim import AdamW

import gc

from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader, random_split

class TextTaskDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = []  # list of (story_idx, frame_idx)

        for story_idx, example in enumerate(self.dataset):
            num_frames = example["frame_count"]
            for frame_idx in range(num_frames):
                self.index.append((story_idx, frame_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        story_idx, frame_idx = self.index[idx]
        example = self.dataset[story_idx]

        image_attributes = parse_gdi_text(example["story"])
        description = image_attributes[frame_idx]["description"]

        return description
    
def trainingLoop(epochs,data_collator,text_dataloader,tokenizer,device):

    model = BertForMaskedLM.from_pretrained("google-bert/bert-base-uncased").to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = len(text_dataloader)
        for step, descriptions in enumerate(text_dataloader):
            # Move the "sentences" to device
            batch = tokenizer(
                descriptions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=100
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            examples = [
                {"input_ids": ids, "attention_mask": mask}
                for ids, mask in zip(batch["input_ids"], batch["attention_mask"])
            ]
            masked_batch = data_collator(examples)
            masked_batch = {k: v.to(device) for k, v in masked_batch.items()}
            outputs = model(**masked_batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if step % 50 == 0:
                print(f"[Epoch {epoch+1}] Step {step}/{num_batches} | Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1}/{epochs}; Avg loss {epoch_loss/len(text_dataloader)}; Latest loss {loss.item()}")
        # torch.save(model.state_dict(), "/content/drive/MyDrive/bert_mlm_finetuned.pth")

if __name__ == "__main__":
    train_dataset = load_dataset("daniel3303/StoryReasoning", split="train")
    test_dataset = load_dataset("daniel3303/StoryReasoning", split="test")
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    text_dataset = TextTaskDataset(train_dataset)
    dataloader = DataLoader(text_dataset, batch_size=32, shuffle=True)

# @title Example text reconstruction task
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    N_EPOCHS = 3
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", padding=True, truncation=True)
    text_dataset = TextTaskDataset(train_dataset)
    text_dataset.index = text_dataset.index[:10000]
    text_dataloader = DataLoader(text_dataset, batch_size=32, shuffle=True)
    tokens = tokenizer(text_dataset[0])
    print(len(tokens["input_ids"]))

    print("Number of descriptions in training set:", len(text_dataset))
    print("Number of batches per epoch:", len(text_dataloader))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,  # 15% of tokens masked (default)
    )
    trainingLoop(N_EPOCHS,data_collator,text_dataloader,tokenizer,device)

