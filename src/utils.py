# @title Setting up google drive to save checkpoints
import torch
import os
import re
from bs4 import BeautifulSoup
from transformers import BertTokenizer
import torchvision.transforms as transforms
import torch.nn as nn
import textwrap
import matplotlib.pyplot as plt
from data import *
# This will prompt you to authorize Google Drive access
device= torch.device('cuda')

# Universal checkpoint handling
# Works in Colab and VS Code

import os
import torch

# CHANGE THIS ONLY
RUNNING_ON_COLAB = False   # True in Colab, False in VS Code

if RUNNING_ON_COLAB:
    from google.colab import drive
    drive.mount('/content/gdrive')
    BASE_DIR = "/content/gdrive/MyDrive"
else:
    # Your local Google Drive synced path (Windows)
    BASE_DIR = r"G:\My Drive"


CHECKPOINT_DIR = os.path.join(BASE_DIR, "DL_Checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_checkpoint_to_drive(model, optimizer, epoch, loss,
                    filename="autoencoder_checkpoint.pth"):

    full_path = os.path.join(CHECKPOINT_DIR, filename)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, full_path)
    print(f"Checkpoint saved: {full_path} (epoch {epoch})")


def load_checkpoint_from_drive(model, optimizer=None,
                    filename="autoencoder_checkpoint.pth",
                    device=None):

    full_path = os.path.join(CHECKPOINT_DIR, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Checkpoint not found: {full_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(full_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)

    print(f"Checkpoint loaded: {full_path} (epoch {epoch})")
    return model, optimizer, epoch, loss


# @title Functions to load images and process data


# This function just extracts the tags from the text, don't get distracted by it.
# I changed this function a bit to fix some bugs
def parse_gdi_text(text):
    """Parse GDI formatted text into structured data"""
    soup = BeautifulSoup(text, 'html.parser')
    images = []

    for gdi in soup.find_all('gdi'):
        # Debug: print what BeautifulSoup sees

        # Method 1: Try to get image attribute directly
        image_id = None
        if gdi.attrs:
            # Check for attributes like 'image1', 'image2', etc.
            for attr_name, attr_value in gdi.attrs.items():
                if 'image' in attr_name.lower():
                    image_id = attr_name.replace('image', '')
                    break

        # Method 2: Extract from the tag string using regex
        if not image_id:
            tag_str = str(gdi)
            match = re.search(r'<gdi\s+image(\d+)', tag_str)
            if match:
                image_id = match.group(1)

        # Method 3: Fallback - use sequential numbering
        if not image_id:
            image_id = str(len(images) + 1)

        content = gdi.get_text().strip()

        # Extract tagged elements using BeautifulSoup directly
        objects = [obj.get_text().strip() for obj in gdi.find_all('gdo')]
        actions = [act.get_text().strip() for act in gdi.find_all('gda')]
        locations = [loc.get_text().strip() for loc in gdi.find_all('gdl')]

        images.append({
            'image_id': image_id,
            'description': content,
            'objects': objects,
            'actions': actions,
            'locations': locations,
            'raw_text': str(gdi)
        })

    return images

# This is an utility function to show images.
# Why do we need to do all this?
def show_image(ax, image, de_normalize = False, img_mean = None, img_std = None):
  """
  De-normalize the image (if necessary) and show image
  """
  if de_normalize:
    new_mean = -img_mean/img_std
    new_std = 1/img_std

    image = transforms.Normalize(
        mean=new_mean,
        std=new_std
    )(image)
  ax.imshow(image.permute(1, 2, 0))

# @title Generate helper fucntion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@torch.no_grad()
def generate(
    model,
    desc_emb,
    obj_emb,
    act_emb,
    tokenizer,
    max_len,
    temperature=0.7
):
    device = desc_emb.device

    # ---------- projections ----------
    desc = model.text_proj(desc_emb)   # [B, S, D]
    obj  = model.obj_proj(obj_emb)
    act  = model.act_proj(act_emb)

    # ---------- gated conditioning ----------
    cond = torch.cat([obj, act], dim=-1)          # [B, S, 2D]
    gate = torch.sigmoid(model.gate_layer(cond))  # [B, S, D]
    desc = desc * gate                             # [B, S, D]

    # ---------- temporal reasoning ----------
    zseq, _ = model.temporal_rnn(desc)             # [B, S, D]

    # ---------- attention ----------
    context = model.attention(zseq)                # [B, D]
    h_last = zseq[:, -1]                            # [B, D]

    z = model.projection(torch.cat([h_last, context], dim=1))  # [B, H]

    # ---------- decoder init ----------
    num_layers = model.text_decoder.gru.num_layers
    h0 = z.unsqueeze(0).repeat(num_layers, 1, 1)

    # ---------- autoregressive decoding ----------
    sos = tokenizer.cls_token_id
    eos = tokenizer.sep_token_id

    generated = torch.tensor([[sos]], device=device)

    for _ in range(max_len):
        logits = model.text_decoder(generated, h0)
        next_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        if next_token.item() == eos:
            break

        generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)




# @title Adding Metrices
# !pip install rouge-score
from rouge_score import rouge_scorer

def compute_rouge_l(pred_texts, ref_texts):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(pred_texts, ref_texts):
        score = scorer.score(ref, pred)["rougeL"].fmeasure
        scores.append(score)
    return sum(scores) / len(scores)

rouge_scorer_single = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def rouge_l_single(pred_text, ref_text):
    return rouge_scorer_single.score(ref_text, pred_text)["rougeL"].fmeasure

# !pip install sentence-transformers
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

semantic_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
semantic_model.eval()

def compute_semantic_similarity(pred_texts, ref_texts):
    sims = []
    with torch.no_grad():
        for pred, ref in zip(pred_texts, ref_texts):
            emb = semantic_model.encode(
                [pred, ref],
                convert_to_tensor=True,
                device=device
            )
            sim = F.cosine_similarity(emb[0], emb[1], dim=0)
            sims.append(sim.item())
    return sum(sims) / len(sims)

# @title Semantic text loss
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

semantic_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
semantic_model.eval()
def semantic_loss(pred_texts, ref_texts):
    with torch.no_grad():  # IMPORTANT: no gradients here
        emb_pred = semantic_model.encode(
            pred_texts,
            convert_to_tensor=True,
            device=device
        )
        emb_ref = semantic_model.encode(
            ref_texts,
            convert_to_tensor=True,
            device=device
        )

        sim = F.cosine_similarity(emb_pred, emb_ref, dim=1)
        loss = 1.0 - sim.mean()

    return loss


# @title Validation functions: To initialize and to visualize the progress
def validation(model, data_loader, criterion, max_gen_len=120):
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    pred_texts = []
    ref_texts  = []
    printed_sample = False

    with torch.no_grad():
        for desc_emb, obj_emb, act_emb, text_target in data_loader:

            desc_emb = desc_emb.to(device)
            obj_emb  = obj_emb.to(device)
            act_emb  = act_emb.to(device)
            text_target = text_target.to(device)

            # ---------- Teacher forcing ----------
            logits = model(desc_emb, obj_emb, act_emb, text_target)

            target_labels = text_target[:, 1:]
            prediction_flat = logits.reshape(-1, tokenizer.vocab_size)
            target_flat = target_labels.reshape(-1)

            loss = criterion(prediction_flat, target_flat)

            total_loss += loss.item() * target_flat.size(0)
            total_tokens += target_flat.size(0)

            # ---------- Generation (for metrics) ----------
            pred_text = generate(
                model,
                desc_emb[:1],
                obj_emb[:1],
                act_emb[:1],
                tokenizer=tokenizer,
                max_len=max_gen_len,
                temperature=0.7
            )

            ref_text = tokenizer.decode(
                text_target[0], skip_special_tokens=True
            )

            pred_texts.append(pred_text)
            ref_texts.append(ref_text)

            # ---------- Print ONE qualitative example ----------
            if not printed_sample:
                printed_sample = True
                import textwrap
                print("\n" + "=" * 80)
                print("TARGET:")
                print(textwrap.fill(ref_text, 100))
                print("\nPREDICTED:")
                print(textwrap.fill(pred_text, 100))
                print("=" * 80 + "\n")

    avg_loss = total_loss / total_tokens
    avg_rouge = compute_rouge_l(pred_texts, ref_texts)

    # semantic similarity averaged properly
    avg_sem = compute_semantic_similarity(pred_texts, ref_texts)

    return avg_loss, avg_rouge, avg_sem


# @title Caching BERT embeddings for train/val/test
from tqdm.auto import tqdm

def cach_data(name, loader,text_autoencoder):
    desc_embs, obj_embs, act_embs, targets = [], [], [], []
    text_autoencoder.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Caching {name}"):
            descriptions, objects, actions, text_target = batch
            descriptions = descriptions.to(device)
            objects = objects.to(device)
            actions = actions.to(device)

            # dynamic dims
            b, seq_len, desc_len = descriptions.shape
            obj_len = objects.shape[2]
            act_len = actions.shape[2]

            # flatten first two dims
            desc_flat = descriptions.view(b * seq_len, desc_len)
            obj_flat  = objects.view(b * seq_len, obj_len)
            act_flat  = actions.view(b * seq_len, act_len)

            # run BERT
            desc_out = text_autoencoder.bert(desc_flat).last_hidden_state[:, 0, :].cpu()
            obj_out  = text_autoencoder.bert(obj_flat).last_hidden_state[:, 0, :].cpu()
            act_out  = text_autoencoder.bert(act_flat).last_hidden_state[:, 0, :].cpu()

            # reshape back
            hidden_dim = desc_out.size(-1)  # should be 768
            desc_out = desc_out.view(b, seq_len, hidden_dim)
            obj_out  = obj_out.view(b, seq_len, hidden_dim)
            act_out  = act_out.view(b, seq_len, hidden_dim)

            desc_embs.append(desc_out)
            obj_embs.append(obj_out)
            act_embs.append(act_out)
            targets.append(text_target.cpu())

    # concatenate all
    desc_embs = torch.cat(desc_embs)
    obj_embs  = torch.cat(obj_embs)
    act_embs  = torch.cat(act_embs)
    targets   = torch.cat(targets)

    torch.save({
        "desc": desc_embs,
        "obj": obj_embs,
        "act": act_embs,
        "target": targets
    }, f"/content/gdrive/MyDrive/BERTcach/cached_bert_embeddings_{name}.pt")

    print(f"✅ Cached {name} BERT features saved successfully!")


# @title Description Task Results
def plotSequancePredcition(train_losses,val_losses,val_rouge_l,val_sem_sim,sequence_predictor,test_dataloader_c,criterion_text):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.show()

    plt.figure()
    plt.plot(val_rouge_l, label="ROUGE-L")
    plt.plot(val_sem_sim, label="Semantic Similarity")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Validation Metrics")
    plt.show()

    test_loss, test_rouge, test_sem = validation(
        sequence_predictor,
        test_dataloader_c,
        criterion_text
    )

    print("TEST SET PERFORMANCE")
    print(f"Loss: {test_loss:.4f}")
    print(f"ROUGE-L: {test_rouge:.4f}")
    print(f"Semantic Similarity: {test_sem:.4f}")
