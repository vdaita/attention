# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, get_scheduler
import os
import torch
from torch.optim import AdamW
import time
from torch import nn, Tensor
import argparse

def generate_beacon_attention_mask_2d(size, window_length=4, direct_window_multiple=1, device=None):
    mask_tensor = torch.zeros((size, size), device=device)
    mask_tensor[:, ::window_length] = 1
    for i in range(size):
        start_index = max(0, i - window_length*direct_window_multiple)
        mask_tensor[i, start_index:i] = 1
        mask_tensor[i, i] = 0
    return mask_tensor.tril()

sep_token = 186 # This corresponds to token: þ in the tokenizer for TinyStories-1m
sep_char = 'þ'

# %%
# Change config here:

parser = argparse.ArgumentParser(
   prog="BeaconEmbeddingTrain",
)
parser.add_argument('-m', '--use-custom-attn-mask', action="store_true", help="Use the custom attention mask (BeaconAttention)")
parser.add_argument('-w', '--window-size', type=int, help="Window Size")

args = parser.parse_args()

use_embedding = args.use_embedding
use_custom_attn_mask = args.use_custom_attn_mask
window_size = args.window_size

print("Window size: ", window_size)
print("Custom attention mask: ", use_custom_attn_mask)
print("Use custom embedding: ", use_embedding)

num_epochs = 1
batch_size = 128
block_size = 128

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# %%
# https://huggingface.co/learn/nlp-course/en/chapter7/6?fw=pt
from torch.nn import CrossEntropyLoss
import torch

def causal_lm_loss(inputs, logits, alpha=1.0):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    mask = (shift_labels != sep_token)
    shift_labels = shift_labels[mask]
    shift_logits = shift_logits[mask]

    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    # Calculate average
    loss = loss_per_sample.mean()
    return loss

# %%
from datasets import load_dataset

def tokenize(element):
    text_with_pause = []
    for i in range(len(element['text'])):
        if (i + 1) % window_size == 0:
            text_with_pause.append(sep_char)
        text_with_pause.append(element['text'][i])
    text_with_pause = ''.join(text_with_pause)

    outputs = tokenizer(
        text_with_pause,
        truncation=True,
        max_length=block_size,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == block_size:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

ds = load_dataset("roneneldan/TinyStories")
tokenized_ds = ds.map(tokenize, batched=True, batch_size=batch_size, num_proc=4, remove_columns=ds["train"].column_names)
tokenized_ds.set_format("torch")

# %%
from torch.utils.data.dataloader import DataLoader

train_dataloader = DataLoader(tokenized_ds["train"], batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(tokenized_ds["validation"], batch_size=batch_size)

# %%
print(len(eval_dataloader))

# %%
weight_decay = 0.1

def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

# %%
model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-1m')

model.to(device)

# %%
model.config

# %%

beacon_attention_mask = generate_beacon_attention_mask_2d(block_size, window_length=window_size, device=device)
# beacon_attention_mask = beacon_attention_mask.unsqueeze(0).repeat(batch_size, 1, 1)

# %%
print("Batch size: ", batch_size)

# %%
print("Attention mask: ", beacon_attention_mask[:10, :10])

optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = num_epochs * len(train_dataloader)

# %%
def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            if use_custom_attn_mask:
                # print("Shape: ", batch["input_ids"].shape, beacon_attention_mask.shape)
                X, Y = batch["input_ids"].shape
                outputs = model(batch["input_ids"], labels=batch["input_ids"], attention_mask=beacon_attention_mask[:X, :Y])
            else:
                outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
    # print(losses)
    loss = torch.mean(torch.Tensor(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()

# %%
from accelerate import Accelerator
import wandb

accelerator = Accelerator() # Logging with wandb here isn't working as expected for some reason
wandb.init(
    project="beacon-attention-1m",
    config={
        "use_custom_attn_mask": use_custom_attn_mask,
        "window_size": window_size,
        "summarization": True
    }
)

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

# %%
model_name = f"{'beacon_embed' if use_embedding else 'no_beacon_embed'}_{'beacon_attn_mask' if use_custom_attn_mask else 'regular_attn_mask'}_window_size_{window_size}_model"
output_dir = f"./models/{model_name}"
if not(os.path.exists(output_dir)):
    os.makedirs(output_dir)

# %%
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# %%
from tqdm import tqdm

gradient_accumulation_steps = 8
eval_steps = 5000

model.train()

completed_steps = 0
step_start_time = time.perf_counter()

eval_loss, perplexity = evaluate()

for epoch in range(num_epochs):
    for step, batch in tqdm(enumerate(train_dataloader, start=1), total=num_training_steps):
        X, Y = batch['input_ids'].shape
        if use_custom_attn_mask:
            # print("Shape: ", batch["input_ids"].shape, beacon_attention_mask.shape)
            logits = model(input_ids=batch["input_ids"], attention_mask=beacon_attention_mask[:X, :Y]).logits
        else:
            logits = model(input_ids=batch["input_ids"]).logits
        loss = causal_lm_loss(batch["input_ids"], logits)

        if step % 100 == 0:
            step_end_time = time.perf_counter()
            train_update = {
                "samples": step * batch_size,
                "steps": completed_steps,
                "loss/train": loss.item(), # * gradient_accumulation_steps,
                "step_time": step_end_time - step_start_time
            }
            accelerator.print(train_update)
            wandb.log(train_update)
            step_start_time = step_end_time

        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)

        if step % gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
        if (step % 1000) == 0:
            eval_loss, perplexity = evaluate()
            eval_update = {"loss/eval": eval_loss, "perplexity": perplexity}
            accelerator.print(eval_update)
            wandb.log(eval_update)
            model.train()
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        
        end_time = time.perf_counter()

# %%
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
wandb.finish()

# %%



