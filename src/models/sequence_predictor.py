# @title Importing the necessary libraries

from data import *
from utils import *

import torch
import torch.nn as nn
import textwrap
import torch.nn.functional as F
import numpy as np
from datasets.fingerprint import random
from transformers import BertTokenizer, BertForMaskedLM

# @title Only Text dataset
class SeqTextPredictionDataset(Dataset):
    def __init__(self, original_dataset, tokenizer,window_size=5, stride=4):
        super(SequencePredictionDataset, self).__init__()
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

# @title A simple attention architecture

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # This "attention" layer learns a query vector
        self.attn = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1) # Over the sequence length

    def forward(self, rnn_outputs):
        # rnn_outputs shape: [batch, seq_len, hidden_dim]

        # Pass through linear layer to get "energy" scores
        energy = self.attn(rnn_outputs).squeeze(2) # Shape: [batch, seq_len]

        # Get attention weights
        attn_weights = self.softmax(energy) # Shape: [batch, seq_len]

        # Apply weights
        # attn_weights.unsqueeze(1) -> [batch, 1, seq_len]
        # bmm with rnn_outputs -> [batch, 1, hidden_dim]
        context = torch.bmm(attn_weights.unsqueeze(1), rnn_outputs)

        # Squeeze to get final context vector
        return context.squeeze(1) # Shape: [batch, hidden_dim]

# @title The main sequence predictor model

class SequencePredictor(nn.Module):
    def __init__(self, text_autoencoder, latent_dim,
                 gru_hidden_dim):
        super(SequencePredictor, self).__init__()
        # --- 1. Encoders ---
        self.text_encoder = text_autoencoder.encoder  
        self.obj_encoder = text_autoencoder.encoder  
        self.act_encoder = text_autoencoder.encoder 

        # --- 2. gate ---
        self.gate_layer = nn.Linear(latent_dim * 2, latent_dim) 

        # --- 3. Temporal Encoder ---
        fusion_dim = latent_dim * 2 # z_conditions fuse + z_text
        self.temporal_rnn = nn.GRU(fusion_dim, latent_dim, batch_first=True)

        # --- 4. Attention ---
        self.attention = Attention(gru_hidden_dim)

        # --- 4. Final Projection ---
        # cat(h, context) -> gru_hidden_dim * 2
        self.projection = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, latent_dim),
            nn.ReLU()
        )
        # --- 5. Decoders ---
        # (These predict the *next* item)
        self.text_decoder = text_autoencoder.decoder

        self.fused_to_h0 = nn.Linear(latent_dim, 32)
        self.fused_to_c0 = nn.Linear(latent_dim, 32)

    def forward(self, obj_seq, act_seq, desc_seq, target_seq):
        batch_size, seq_len, _ = desc_seq.shape

        # --- 1 & 2: Run Static Encoders over the sequence ---
        # Reshape for text_encoders
        desc_flat = desc_seq.view(batch_size * seq_len, -1) # -1 infers text_len
        obj_flat = obj_seq.view(batch_size * seq_len, -1)
        act_flat = act_seq.view(batch_size * seq_len, -1)

        # Run encoders
        _, desc_hidden, _ = self.text_encoder(desc_flat)
        _, obj_hidden, _ = self.obj_encoder(obj_flat)
        _, act_hidden, _ = self.act_encoder(act_flat)

        desc_hidden = desc_hidden.squeeze(0)
        obj_hidden = obj_hidden.squeeze(0)
        act_hidden = act_hidden.squeeze(0)

        # Gating
        cond = torch.cat([obj_hidden, act_hidden], dim=-1) # concatinate objects and actions as one feature
        gate = torch.sigmoid(self.gate_layer(cond)) # gate 
        conditioned_desc = desc_hidden * gate  # apply on desc
        conditioned_seq = conditioned_desc.view(batch_size, seq_len, -1) 

        # Temporal 
        # zseq shape: [b, s, gru_hidden]
        # h    shape: [1, b, gru_hidden]
        zseq, h = self.temporal_rnn(conditioned_seq)
        h = h.squeeze(0)

        # --- 4. Attention ---
        context = self.attention(zseq)

        # --- 5. Final Prediction Vector (z) ---
        z = self.projection(torch.cat((h, context), dim=1)) # Shape: [b, joint_latent_dim]

        # --- 6. Decode (Predict pk) ---

        h0 = self.fused_to_h0(z).unsqueeze(0)
        c0 = self.fused_to_c0(z).unsqueeze(0)

        decoder_input = target_seq[:, :,:-1].squeeze(1)

        # 3. Run the decoder *once* on the entire sequence.
        # It takes the encoder's final state (hidden, cell)
        # and the full "teacher" sequence (decoder_input).
        predicted_text_logits_k, _, _ = self.text_decoder(decoder_input, h0, c0)
        # return pred_image_content, pred_image_context, predicted_text_logits_k,h0, c0
        return  predicted_text_logits_k,h0, c0
    
def trainingLoop(sequence_predictor,train_dataloader,tokenizer,N_EPOCHS):
    # @title Training tools
    criterion_text = nn.CrossEntropyLoss(ignore_index=tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
    optimizer = torch.optim.Adam(sequence_predictor.parameters(), lr=0.001)

    sequence_predictor.train()
    losses = []
    for epoch in range(N_EPOCHS):

        running_loss = 0.0
        for descriptions, objects, actions, text_target in train_dataloader:
            # Send images and tokens to the GPU
            descriptions = descriptions.to(device)
            act_seq = actions.to(device)
            obj_seq = objects.to(device)
            text_target = text_target.to(device)

            # Predictions from our model
            predicted_text_logits, _, _ = sequence_predictor(descriptions, obj_seq, act_seq, text_target)
            # Computing losses

            # Loss function for the text prediction
            prediction_flat = predicted_text_logits.reshape(-1, tokenizer.vocab_size)
            target_labels = text_target.squeeze(1)[:, 1:] # Slice to get [8, 119]
            target_flat = target_labels.reshape(-1)
            loss = criterion_text(prediction_flat, target_flat)
            # Combining the losses
            # loss = loss_im + loss_text + 0.2*loss_context
            # Optimizing
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * descriptions.size(0)

        # checking model performance on validation set
        sequence_predictor.eval()
        print("Validation on training dataset")
        print( "----------------")
        validation( sequence_predictor, train_dataloader )
        print("Validation on validation dataset")
        print( "----------------")
        validation( sequence_predictor, val_dataloader)
        sequence_predictor.train()

        # scheduler.step()
        epoch_loss = running_loss / len(train_dataloader.dataset)
        losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{N_EPOCHS}], Loss: {epoch_loss:.4f}')

        if epoch % 5 == 0:
            save_checkpoint_to_drive(sequence_predictor, optimizer, epoch, epoch_loss, filename=f"sequence_predictor.pth")
    return losses


def __init__():
    N_EPOCHS=25
    latent_dim = 32  
    gru_hidden_dim = 32
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    sequenceDataLoader=0

    text_autoencoder = BertForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
    text_autoencoder.load_state_dict(
        torch.load("/content/drive/MyDrive/bert_mlm_finetuned.pth", map_location=device)
    )
    for param in text_autoencoder.parameters():
        param.requires_grad = False
    text_autoencoder.to(device)
    text_autoencoder.eval()

    sequence_predictor = SequencePredictor(
        text_autoencoder=text_autoencoder,
        latent_dim=latent_dim,
        gru_hidden_dim=gru_hidden_dim
    ).to(device)

    losses=trainingLoop(sequence_predictor,sequenceDataLoader,tokenizer,N_EPOCHS)
    # Do better plots
    plt.plot(losses)
    plt.show()
