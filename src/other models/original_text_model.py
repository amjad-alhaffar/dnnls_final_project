# the pretrained model for the text autoencoder using LSTM   
# @title The text autoencoder (Seq2Seq)

import torch.nn as nn
from datasets import load_dataset
from transformers import BertTokenize

from torch.cuda.amp import autocast, GradScaler
import gc
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from utils import parse_gdi_text
from torch.utils.data import DataLoader
from transformers import BertTokenizer

class EncoderLSTM(nn.Module):
    """
      Encodes a sequence of tokens into a latent space representation.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class DecoderLSTM(nn.Module):
    """
      Decodes a latent space representation into a sequence of tokens.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.out = nn.Linear(hidden_dim, vocab_size) # Should be hidden_dim

    def forward(self, input_seq, hidden, cell):
        embedded = self.embedding(input_seq)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.out(output)
        return prediction, hidden, cell

# We create the basic text autoencoder (a special case of a sequence to sequence model)
class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        # input_seq and target_seq are both your 'input_ids'
        # 1. Encode the input sequence
        _enc_out, hidden, cell = self.encoder(input_seq)

        # 2. Create the "shifted" decoder input for teacher forcing.
        # We want to predict target_seq[:, 1:]
        # So, we feed in target_seq[:, :-1]
        # (i.e., feed "[SOS], hello, world" to predict "hello, world, [EOS]")
        decoder_input = target_seq[:, :-1]

        # 3. Run the decoder *once* on the entire sequence.
        # It takes the encoder's final state (hidden, cell)
        # and the full "teacher" sequence (decoder_input).
        predictions, _hidden, _cell = self.decoder(decoder_input, hidden, cell)

        # predictions shape will be (batch_size, seq_len-1, vocab_size)
        return predictions
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



def tainingLoop(text_dataloader,text_autoencoder,tokenizer):
    optimizer = torch.optim.Adam(text_autoencoder.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
    scaler = GradScaler()  # helps stabilize mixed precision training
    N_EPOCHS = 25

    for epoch in range(N_EPOCHS):
        text_autoencoder.train()
        epoch_loss = 0.0
        for step, descriptions in enumerate(text_dataloader):
            batch = tokenizer(
                descriptions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(device)

            input_ids = batch["input_ids"]
            optimizer.zero_grad()
            # Mixed precision forward + backward
            with autocast():
                outputs = text_autoencoder(input_ids, input_ids)
                loss = loss_fn(
                    outputs.reshape(-1, tokenizer.vocab_size),
                    input_ids[:, 1:].reshape(-1)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if step % 100 == 0:
                print(f"[Epoch {epoch+1}] Step {step} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(text_dataloader)
        print(f"Epoch {epoch+1}/{N_EPOCHS} | Avg loss: {avg_loss:.4f}")
        # if (epoch + 1) % 5 == 0:
        #     torch.save(text_autoencoder.state_dict(), f"/content/seq2seq-epoch-{epoch+1}.pth")
        #     print("saved")


if __name__ == "__main__":
    # @title Example text reconstruction task
    train_dataset = load_dataset("daniel3303/StoryReasoning", split="train")
    test_dataset = load_dataset("daniel3303/StoryReasoning", split="test")
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    emb_dim = 32
    latent_dim = 32
    num_layers = 1
    dropout = 0.1
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", padding=True, truncation=True)
    text_dataset = TextTaskDataset(train_dataset)
    text_dataset.index = text_dataset.index[:10000]
    text_dataloader = DataLoader(text_dataset, batch_size=32, shuffle=True)
    tokens = tokenizer(text_dataset[0])
    print(len(tokens["input_ids"]))

    print("Number of descriptions in training set:", len(text_dataset))
    print("Number of batches per epoch:", len(text_dataloader))

    encoder = EncoderLSTM(tokenizer.vocab_size, emb_dim, latent_dim, num_layers, dropout).to(device)
    decoder = DecoderLSTM(tokenizer.vocab_size, emb_dim, latent_dim, num_layers, dropout).to(device)
    text_autoencoder = Seq2SeqLSTM(encoder, decoder).to(device)
    tainingLoop(text_dataloader,text_autoencoder,tokenizer)
    # save_checkpoint_to_drive(text_autoencoder, optimizer, 3*N_EPOCHS, loss, filename = "text_autoencoder.pth")
