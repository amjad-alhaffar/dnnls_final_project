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
class TextDecoderGRU(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        hidden_dim,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.gru = nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, h0):
        emb = self.embedding(input_ids)      # [B, T, E]
        out, _ = self.gru(emb, h0)            # [B, T, H]
        out = self.dropout(out)
        logits = self.fc(out)                 # [B, T, V]
        return logits
# @title Bert wrapper
class BertAutoencoderWrapper(nn.Module):
    def __init__(self, bert_mlm_model):
        super().__init__()
        self.bert = bert_mlm_model.bert        # encoder
        self.cls = bert_mlm_model.cls          # MLM head

    def encoder(self, input_ids):
        """
        Returns (dummy_output, hidden, cell)
        We'll use the [CLS] token embedding as 'hidden'.
        """
        outputs = self.bert(input_ids)
        hidden_state = outputs.last_hidden_state
        cls_token = hidden_state[:, 0, :]
        return None, cls_token.unsqueeze(0), None

    def decoder(self, input_ids, h0, c0):
        outputs = self.bert(input_ids)
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.cls(sequence_output)
        # return logits in same format as before
        return prediction_scores, h0, c0

class SequencePredictorCached(nn.Module):
    def __init__(self, latent_dim, vocab_size):
        super().__init__()

        # projections
        self.text_proj = nn.Linear(768, latent_dim)
        self.obj_proj  = nn.Linear(768, latent_dim)
        self.act_proj  = nn.Linear(768, latent_dim)


        # Gating
        self.gate_layer = nn.Linear(latent_dim * 2, latent_dim)

        # Temporal
        self.temporal_rnn = nn.GRU(latent_dim, latent_dim, batch_first=True)

        # Attention over frames
        self.attention = Attention(latent_dim)

        # Final fusion
        self.projection = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU()
        )

        # decoder
        self.text_decoder = TextDecoderGRU(
            vocab_size=vocab_size,
            emb_dim=latent_dim,
            hidden_dim=latent_dim,
            num_layers=2,
            dropout=0.1
        )

    def forward(self, desc_emb, obj_emb, act_emb, target_seq):

        # Project BERT CLS per frame
        desc = self.text_proj(desc_emb)  # [B, S, D]
        obj  = self.obj_proj(obj_emb)
        act  = self.act_proj(act_emb)

        # conditioning
        cond = torch.cat([obj, act], dim=-1)     # [B, S, 2D]
        gate = torch.sigmoid(self.gate_layer(cond))
        desc = desc * gate                       # [B, S, D]

        # Temporal
        zseq, _ = self.temporal_rnn(desc)        # [B, S, D]

        # Attention
        context = self.attention(zseq)           # [B, D]
        h_last  = zseq[:, -1]                    # [B, D]

        # Fuse
        z = self.projection(torch.cat([h_last, context], dim=1))

        # Decoder init
        h0 = z.unsqueeze(0).repeat(
            self.text_decoder.gru.num_layers, 1, 1
        )

        # Teacher forcing
        decoder_input = target_seq[:, :-1]
        logits = self.text_decoder(decoder_input, h0)

        return logits
# @title Load cached embeddings to speed the training
class CachedSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, cache_path):
        data = torch.load(cache_path)
        self.desc = data["desc"]
        self.obj = data["obj"]
        self.act = data["act"]
        self.target = data["target"]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return (
            self.desc[idx],
            self.obj[idx],
            self.act[idx],
            self.target[idx]
        )

def trainingLoop(sequence_predictor,train_dataloader_c,val_dataloader_c,tokenizer,criterion_text,scheduler,N_EPOCHS):
    # @title Training loop for the sequence predictor
    SEMANTIC_START_EPOCH = 10
    SEMANTIC_WEIGHT = 0.1

    start_epoch = 0

    try:
        sequence_predictor, optimizer, last_epoch, _ = load_checkpoint_from_drive(
            sequence_predictor,
            optimizer,
            filename=f"sequence_predictor_decoder{start_epoch}.pth"
        )
        start_epoch = last_epoch + 1
    except FileNotFoundError:
        print("Starting from scratch")

    sequence_predictor.train()
    train_losses = []
    val_losses   = []

    val_rouge_l  = []
    val_sem_sim  = []
    print('haha')
    print(len(train_dataloader_c))
    for epoch in range(start_epoch, N_EPOCHS):
        running_loss = 0.0
        for step,(descriptions, objects, actions, text_target)  in enumerate(train_dataloader_c):
            # Send images and tokens to the GPU
            descriptions = descriptions.to(device)
            act_seq = actions.to(device)
            obj_seq = objects.to(device)
            text_target = text_target.to(device)

            # Predictions from our model
            predicted_text_logits = sequence_predictor(
                descriptions,obj_seq, act_seq , text_target
            )
            # Computing losses

            # Loss function for the text prediction
            prediction_flat = predicted_text_logits.reshape(-1, tokenizer.vocab_size)
            target_labels = text_target.squeeze(1)[:, 1:] # Slice to get [8, 119]
            target_flat = target_labels.reshape(-1)
            loss = criterion_text(prediction_flat, target_flat)

            # ---- semantic auxiliary loss after 10 epochs ----
            if epoch >= SEMANTIC_START_EPOCH:
                was_training = sequence_predictor.training
                sequence_predictor.eval()
                with torch.no_grad():
                    pred_text = generate(
                        sequence_predictor,
                        descriptions[:1],
                        obj_seq[:1],
                        act_seq[:1],
                        tokenizer=tokenizer,
                        max_len=120,
                        temperature=0.7
                    )

                if was_training:
                    sequence_predictor.train()               # restore state

                ref_text = tokenizer.decode(
                    text_target[0],
                    skip_special_tokens=True
                )

            sem_loss = semantic_loss([pred_text], [ref_text])

            loss = loss + SEMANTIC_WEIGHT * sem_loss
            # Optimizing
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * descriptions.size(0)

            sequence_predictor.eval()

            avg_train_loss = running_loss / len(train_dataloader_c.dataset)
            train_losses.append(avg_train_loss)

            # ---------------- VALIDATION ----------------
            sequence_predictor.eval()

            val_loss, rouge_l_val, sem_sim_val = validation(
                sequence_predictor,
                val_dataloader_c,
                criterion_text
            )

            val_losses.append(val_loss)
            val_rouge_l.append(rouge_l_val)
            val_sem_sim.append(sem_sim_val)

            print(
                f"Epoch {epoch+1:02d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"ROUGE-L: {rouge_l_val:.4f} | "
                f"Semantic: {sem_sim_val:.4f}"
            )

            scheduler.step(val_loss)
            sequence_predictor.train()

            if (epoch + 1) % 5 == 0:
                save_checkpoint_to_drive(
                    sequence_predictor,
                    optimizer,
                    epoch + 1,
                    avg_train_loss,
                    filename=f"sequence_predictor_decoder{epoch+1}.pth"
                )
    plotSequancePredcition(train_losses,val_losses,val_rouge_l,val_sem_sim,sequence_predictor,val_dataloader_c,criterion_text)


def __init__():
    # @title For the Sequence prediction task
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased",  padding=True, truncation=True)
    sp_train_dataset = SeqTextPredictionDataset(train_dataset, tokenizer, window_size=5, stride=3)
    sp_test_dataset  = SeqTextPredictionDataset(test_dataset,  tokenizer, window_size=5, stride=3)
    print(len(sp_train_dataset))
    print(len(sp_test_dataset))
    # Let's do things properly, we will also have a validation split
    # Split the training dataset into training and validation sets
    train_size = int(0.8 * len(sp_train_dataset))
    val_size = len(sp_train_dataset) - train_size
    train_subset, val_subset = random_split(sp_train_dataset, [train_size, val_size])
    print(sp_train_dataset)
    print(sp_test_dataset)

    # test_dataloader = DataLoader(sp_test_dataset, batch_size=32, shuffle=False)

    # @title Text autoencoder before caching Bert

    from transformers import BertTokenizer, BertForMaskedLM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    N_EPOCHS=50
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    criterion_text = nn.CrossEntropyLoss(ignore_index=tokenizer.convert_tokens_to_ids(tokenizer.pad_token))


    bert_mlm  = BertForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
    if RUNNING_ON_COLAB:
        bert_mlm.load_state_dict(
            torch.load("/content/gdrive/MyDrive/bert_mlm_finetuned.pth", map_location=device)
        )
    else:
        bert_mlm.load_state_dict(
            torch.load(r"G:\My Drive\bert_mlm_finetuned.pth", map_location=device)
        )

    text_autoencoder = BertAutoencoderWrapper(bert_mlm).to(device)
    # freeze
    for param in text_autoencoder.parameters():
        param.requires_grad = False
    bert_trainable = sum(p.numel() for p in text_autoencoder.parameters() if p.requires_grad)
    print(f"BERT trainable params: {bert_trainable}")

    text_autoencoder.eval()
    # no need to run this part we have the files saved on drive
    # cach_data("train", train_dataloader,text_autoencoder)
    # cach_data("val", val_dataloader,text_autoencoder)
    # cach_data("test", test_dataloader,text_autoencoder)

    # @title Main pipline cached

    # load cached data first 
    cached_dataset = CachedSequenceDataset("/content/gdrive/MyDrive/BERTcach/cached_bert_embeddings_train.pt")
    cached_dataset_val = CachedSequenceDataset("/content/gdrive/MyDrive/BERTcach/cached_bert_embeddings_val.pt")
    # cached_dataset_test = CachedSequenceDataset("/content/gdrive/MyDrive/BERTcach/cached_bert_embeddings_test.pt")
    train_dataloader_c = DataLoader(cached_dataset, batch_size=64, shuffle=True)
    val_dataloader_c = DataLoader(cached_dataset_val, batch_size=32, shuffle=True)
    # test_dataloader_c = DataLoader(cached_dataset_test, batch_size=16, shuffle=False)
    sequence_predictor = SequencePredictorCached(
        latent_dim=128,
        vocab_size=tokenizer.vocab_size
    ).to(device)

    # Print model size
    predictor_trainable = sum(p.numel() for p in sequence_predictor.parameters() if p.requires_grad)
    print(f"Predictor trainable params: {predictor_trainable}")

    optimizer = torch.optim.Adam(sequence_predictor.parameters(),  lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    losses=trainingLoop(sequence_predictor,train_dataloader_c,val_dataloader_c,tokenizer,criterion_text,scheduler,N_EPOCHS)
    plotSequancePredcition(train_losses,val_losses,val_rouge_l,val_sem_sim,sequence_predictor,test_dataloader_c,criterion_text)
