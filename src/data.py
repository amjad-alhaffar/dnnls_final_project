from utils import parse_gdi_text, show_image
from datasets import load_dataset
import torchvision.transforms as transforms
from transformers import BertTokenizer
import torch
import torchvision.transforms.functional as FT
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


# @title Only Text dataset
class SeqTextPredictionDataset(Dataset):
    def __init__(self, original_dataset, tokenizer,window_size=5, stride=4):
        super(SeqTextPredictionDataset, self).__init__()
        self.dataset = original_dataset
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride
        self.windows = []
        for story_idx, example in enumerate(self.dataset):
            num_frames = example["frame_count"]
            if num_frames < window_size:
                continue  # skip very short stories

            # sliding windows with given stride
            for start in range(0, num_frames - window_size + 1, self.stride):
                self.windows.append((story_idx, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
      """
      Selects a 5 frame sequence from the dataset. Sets 4 for training and the last one
      as a target.
      """
      story_idx, start = self.windows[idx]
      example = self.dataset[story_idx]

      num_frames = example["frame_count"]
      # frames = example["images"]
      image_attributes = parse_gdi_text(example["story"])

      description_list = []
      obj_list = []
      act_list = []

      for offset in range(self.window_size - 1):
        frame_idx = start + offset
        # Potential experiments: Try using the other attributes in your training
        objects = image_attributes[frame_idx]["objects"]
        actions = image_attributes[frame_idx]["actions"]
        description = image_attributes[frame_idx]["description"]
        # We need to return the tokens for NLP
        input_ids =  self.tokenizer(
          description,
          return_tensors="pt",
          padding="max_length",
          truncation=True,
          max_length=120
        ).input_ids
        object_ids =  self.tokenizer(
          ", ".join(objects),
          return_tensors="pt",
          padding="max_length",
          truncation=True,
          max_length=20
        ).input_ids
        action_ids =  self.tokenizer(
          ", ".join(actions),
          return_tensors="pt",
          padding="max_length",
          truncation=True,
          max_length=20
        ).input_ids

        description_list.append(input_ids.squeeze(0))
        obj_list.append(object_ids.squeeze(0))
        act_list.append(action_ids.squeeze(0))

      target_frame_idx = start + (self.window_size - 1)
      text_target = image_attributes[target_frame_idx]["description"]
      target_ids = self.tokenizer(
          text_target,
          return_tensors="pt",
          padding="max_length",
          truncation=True,
          max_length=120).input_ids.squeeze(0)

      description_tensor = torch.stack(description_list)
      obj_tensor= torch.stack(obj_list)
      act_tensor= torch.stack(act_list)

      return (
              description_tensor,
              obj_tensor,
              act_tensor,
              target_ids,
              )


# @title Main dataset
class SequencePredictionDataset(Dataset):
    def __init__(self, original_dataset, tokenizer,window_size=5, stride=4):
        super(SequencePredictionDataset, self).__init__()
        self.dataset = original_dataset
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride
        # Potential experiments: Try other transforms!
        self.transform = transforms.Compose([
          transforms.Resize((60, 125)),# Reasonable size based on our previous analysis
          transforms.ToTensor(), # HxWxC -> CxHxW
        ])
        self.windows = []
        for story_idx, example in enumerate(self.dataset):
            num_frames = example["frame_count"]
            if num_frames < window_size:
                continue  # skip very short stories

            # sliding windows with given stride
            for start in range(0, num_frames - window_size + 1, self.stride):
                self.windows.append((story_idx, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
      """
      Selects a 5 frame sequence from the dataset. Sets 4 for training and the last one
      as a target.
      """
      story_idx, start = self.windows[idx]
      example = self.dataset[story_idx]
      frames = example["images"]
      image_attributes = parse_gdi_text(example["story"])

      frame_tensors = []
      description_list = []
      obj_list = []
      act_list = []

      for offset in range(self.window_size - 1): 
        frame_idx = start + offset
        image = FT.equalize(frames[frame_idx])
        input_frame = self.transform(image)
        frame_tensors.append(input_frame)
        # Potential experiments: Try using the other attributes in your training
        objects = image_attributes[frame_idx]["objects"]
        actions = image_attributes[frame_idx]["actions"]

        description = image_attributes[frame_idx]["description"]
        # We need to return the tokens for NLP
        input_ids =  self.tokenizer(
          description,
          return_tensors="pt",
          padding="max_length",
          truncation=True,
          max_length=120
        ).input_ids
        object_ids =  self.tokenizer(
          ", ".join(objects),
          return_tensors="pt",
          padding="max_length",
          truncation=True,
          max_length=20
        ).input_ids
        action_ids =  self.tokenizer(
          ", ".join(actions),
          return_tensors="pt",
          padding="max_length",
          truncation=True,
          max_length=20
        ).input_ids

        description_list.append(input_ids.squeeze(0))
        obj_list.append(object_ids.squeeze(0))
        act_list.append(action_ids.squeeze(0))

      target_frame_idx = start + (self.window_size - 1)
      image_target = FT.equalize(frames[target_frame_idx])
      image_target = self.transform(image_target)
      text_target = image_attributes[target_frame_idx]["description"]
      object_target = image_attributes[target_frame_idx]["objects"]
      action_target = image_attributes[target_frame_idx]["actions"]

      target_ids = self.tokenizer(
          text_target,
          return_tensors="pt",
          padding="max_length",
          truncation=True,
          max_length=120).input_ids.squeeze(0)
      target_obj =  self.tokenizer(
          ", ".join(object_target),
          return_tensors="pt",
          padding="max_length",
          truncation=True,
          max_length=20
        ).input_ids.squeeze(0)
      target_action =  self.tokenizer(
          ", ".join(action_target),
          return_tensors="pt",
          padding="max_length",
          truncation=True,
          max_length=20
        ).input_ids.squeeze(0)

      sequence_tensor = torch.stack(frame_tensors)  
      description_tensor = torch.stack(description_list) 
      obj_tensor= torch.stack(obj_list)    
      act_tensor= torch.stack(act_list)

      return (sequence_tensor, 
              description_tensor, 
              obj_tensor,
              act_tensor,
              image_target, 
              target_ids,
              target_obj,
              target_action
              ) 

# @title Dataset for image autoencoder task
class AutoEncoderTaskDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
          transforms.Resize((240, 500)),# Reasonable size based on our previous analysis
          transforms.ToTensor(), # HxWxC -> CxHxW
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
      num_frames = self.dataset[idx]["frame_count"]
      frames = self.dataset[idx]["images"]

      # Pick a frame at random
      frame_idx = torch.randint(0, num_frames-1, (1,)).item()
      input_frame = self.transform(frames[frame_idx]) # Input to the autoencoder

      return input_frame # Returning the image





# Count frames from dataset

if __name__ == "__main__":
  
  train_dataset = load_dataset("daniel3303/StoryReasoning", split="train")
  test_dataset = load_dataset("daniel3303/StoryReasoning", split="test")
  tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased",  padding=True, truncation=True)
  sp_train_dataset = SequencePredictionDataset(train_dataset, tokenizer, window_size=5, stride=3) # Instantiate the train dataset
  sp_test_dataset = SequencePredictionDataset(test_dataset, tokenizer, window_size=5, stride=3) # Instantiate the test dataset
  # Let's do things properly, we will also have a validation split
  # Split the training dataset into training and validation sets
  train_size = int(0.8 * len(sp_train_dataset))
  val_size = len(sp_train_dataset) - train_size
  train_subset, val_subset = random_split(sp_train_dataset, [train_size, val_size])

  # Instantiate the dataloaders
  train_dataloader = DataLoader(train_subset, batch_size=8, shuffle=True)
  # We will use the validation set to visualize the progress.
  val_dataloader = DataLoader(val_subset, batch_size=4, shuffle=True)
  test_dataloader = DataLoader(sp_test_dataset, batch_size=4, shuffle=False)
  frame_counter_train = Counter()
  for example in train_dataset:
      num_frames = example["frame_count"]
      frame_counter_train[num_frames] += 1

  min_len = min(5)  
  max_len = max(frame_counter_train.keys())  
  lengths = list(range(min_len, max_len + 1))
  counts = [frame_counter_train.get(l, 0) for l in lengths]
  # Plot
  plt.figure(figsize=(10, 4))
  plt.bar(lengths, counts)
  plt.xticks(lengths)
  plt.xlabel("Number of frames in story")
  plt.ylabel("Number of stories")
  plt.title(f"Story length distribution ({min_len}–{max_len} frames)")
  plt.grid(axis="y", linestyle="--", alpha=0.4)
  plt.show()
