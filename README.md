# DNNLS Final Assessment  
**Author:** Amjad Alhaffar

---

## Introduction and Problem Statement

This repository contains the final assessment for the **Deep Neural Networks and Learning Systems (DNNLS)** course. The goal of this project is to change how the **baseline storytelling model** is predicting the story, instead of predciting a frame with image and text from a sequence of images and text as one, I will focus on 2 tasks .improving the quality of the desciption (text and image), this will be done by 2 seperate stages,
- Improving the quality of scene description text predicted from a sequence of text.
- Generate image from the predicted description, using text-to-image model.
by seperating those stages,first focuses on learning semantics and narretive coherence of the text,while the second stage focuses on the visual style consistency and to make the text more aligned with its image.

The task focuses on **story reasoning**, where a model must understand causal and temporal relationships between events in short stories and generate consistent narrative outcomes.

We use the **StoryReasoning** dataset for training and evaluation:

- Dataset: https://huggingface.co/datasets/daniel3303/StoryReasoning

### Problem Definition

Given a baseline sequence-to-sequence storytelling model, this project introduces several data-level and architectural changes:
- Apply overlapping stories (Sliding window) to extract more stories from the dataset.
- Train the given LSTM text autoencoder.
- Fine tune Bert MLM text autoencoder on our dataset.
- apply objects and actions as conditioning signals to improve the description quality.
- Generate realistic images from predicted descriptions using a diffusion-based model.

Performance for the models is evaluated using both quantitative metrics and human evalutaion.
---

## Methods

### Model Architecture Overview

The final pipline of our architecure:
Sliding window frames → Text features → Sequence-to-Text Model → Text Prompt → Diffusion Image Generation

### Sliding window frames
To increase the training samples, a sliding window approach is applied through the stories.
example: story contains 7 frames can have 2 stories instead of 1.
with the ability to contole the number of frames each story + number of stride.
#### Code Snippet 
```python
for story_idx, example in enumerate(self.dataset):
    num_frames = example["frame_count"]
    if num_frames < window_size:
        continue  # skip very short stories
    # sliding windows with given stride
    for start in range(0, num_frames - window_size + 1, self.stride):
        self.windows.append((story_idx, start))
```
## Results
- data size before & after `results/images/dataset.jpeg`

### Text Auto-encoders Bert MLM & LSTM:
To improve text embeddings, the autoencoders was traied and fine tuned on 1k descriptions with 32 latent space.
Bert MLM: fine tuned for 3 epochs.
Original LSTM model: trained for 25 epochs.

finally Bert was used as a text encoder in our pipline. because it has a lower loss after tuning on our data set.
## Results
- bert training results `results/images/BERT_training.jpg`
- LSTM training results `results/images/lstm_training.jpg`


### Sequence-to-Text Prediction Model:
A sequence prediction model is trained to predict a the description conditioned on the previous four frames.
To better focus on textual reasoning and temporal coherence, the architecture was redesigned into a text-centric sequence predictor, with the following key changes:
- Removal of Image Prediction part, because it is handled outside this model
- BERT text encoding: only the encoder part (CLS embedding) was used for Frame descriptions, objects, and actions.
"theese embeddings are cached to reduce trainig time and memory"
- Multi Conditioning with Gating: objects and actions using gated fusion instead of concatentaion.
- GRU-Based Text Decoder: This part was added as a replacement for Bert MLM decoder as the final latent vector decoder.
- semantic loss was used to assist 
- ROUGE-L and Cosine similarity was used to assist the validation 
#### Code Snippet 
```python
def forward(self, desc_emb, obj_emb, act_emb, target_seq):

    # Project BERT CLS per frame
    desc = self.text_proj(desc_emb)  
    obj  = self.obj_proj(obj_emb)
    act  = self.act_proj(act_emb)

    # conditioning
    cond = torch.cat([obj, act], dim=-1)     
    gate = torch.sigmoid(self.gate_layer(cond))
    desc = desc * gate                       

    # Temporal
    zseq, _ = self.temporal_rnn(desc)        

    # Attention
    context = self.attention(zseq)           
    h_last  = zseq[:, -1]                    

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

class TextDecoderGRU(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        hidden_dim,
        num_layers=2,
        dropout=0.3
    ):
...
```
## Results
- training loss `results/images/description loss.txt`
- training metrics `results/images/description metrics.txt`
- test sample `results/images/final prediction.txt`

### Attempted Prefix Tuning (Exploratory)

Since BERT’s masked language modeling head is not autoregressive, it could not function as a decoder for the sequence-to-text generation task. As a workaround, prefix tuning was tested to adapt BERT for generation. However, this approach failed to produce a readable text.Consequently, I transitioned to a hybrid architecture where BERT works as an encoder, and a trainable GRU-based decoder was added to handle autoregressive text generation.

- code 'prefix tuning attempt.ipynb
- results `results/images/prefix_tuning attempt with BERT.png`


### Text-to-Image Generation with LoRA:
For image generation, Stable Diffusion 1.5 is fine-tuned using Low-Rank Adaptation (LoRA) on image–caption pairs derived from the dataset.
fine tuning was done to get a realistc style images, to make it close as much as possible to the dataset.

## Results
- training log `results/images/Lora training log.txt`

---
### Conclustion
### Results and Discussion

The redesigned sequence-to-text model produced descriptions that were generally more coherent and semantically aligned with the story context compared to the baseline. Conditioning on objects and actions helped reduce vague or generic outputs, while the GRU-based decoder enabled more stable autoregressive generation.

However, the model occasionally generated repetitive or overly generic phrases, particularly in scenes with limited visual diversity. While automatic metrics showed modest improvements, qualitative inspection revealed clearer gains in narrative consistency than raw scores alone.

## Future Work

- External knowledge integration
- Longer story contexts
- Multi-head reasoning states
- Human evaluation studies
