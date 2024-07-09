# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, get_scheduler
import os
import torch
from torch.optim import AdamW
import time
from torch import nn, Tensor
import argparse

class BeaconEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, vocab_size: int, n_embed: int, window_length: int, use_normal_initialization: bool,*args, **kwargs):
        super().__init__()
        self.n_embed = n_embed
        self.b_embed = nn.Parameter(torch.empty(n_embed), requires_grad=True)
        self.window_length = window_length
        self.embedding = embedding
        self.use_normal_initialization = use_normal_initialization
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.use_normal_initialization:
            nn.init.normal_(self.b_embed)
        else:
            nn.init.zeros_(self.b_embed)

    def forward(self, input: Tensor) -> Tensor:
        B, N = input.shape
        regular_embedding = self.embedding(input)
        beacon_tensor = torch.zeros((B, N, self.n_embed))
        beacon_tensor[:, ::self.window_length, :] = self.b_embed
        return regular_embedding + beacon_tensor

def generate_beacon_attention_mask_2d(size, window_length=4, direct_window_multiple=1, device=None):
    mask_tensor = torch.zeros((size, size), device=device)
    mask_tensor[::window_length, :] = 1
    for i in range(size):
        start_index = max(0, i - window_length*direct_window_multiple)
        mask_tensor[i, start_index:i] = 1
    return mask_tensor.tril()

# %%
# Change config here:

parser = argparse.ArgumentParser(
    prog="BeaconEmbeddingTrain",
)
parser.add_argument('-e', '--use-embedding', action="store_true")
parser.add_argument('-m', '--use-custom-attn-mask', action="store_true")
parser.add_argument('-n', '--use-normal-initialization', action="store_true")
parser.add_argument('-w', '--window-size', type=int)

args = parser.parse_args()

use_embedding = args.use_embedding
use_custom_attn_mask = args.use_custom_attn_mask
use_normal_initialization = args.use_normal_initialization
window_size = args.window_size

print("Window size: ", window_size)
print("Normal initialization: ", use_normal_initialization)
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
    outputs = tokenizer(
        element["text"],
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
model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-3M')

model.to(device)

# %%
model.config

# %%

if use_embedding:
    beacon_embedding = BeaconEmbedding(embedding=model.get_input_embeddings(), vocab_size=model.config.vocab_size, n_embed=model.config.hidden_size, window_length=window_size, use_normal_initialization=use_normal_initialization)
    model.set_input_embeddings(beacon_embedding)

beacon_attention_mask = generate_beacon_attention_mask_2d(block_size, window_length=window_size, device=device)
beacon_attention_mask = beacon_attention_mask.unsqueeze(0).repeat(batch_size, 1, 1)


optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = num_epochs * len(train_dataloader)

# %%
print("Batch size: ", batch_size)

# %%
print("Attention mask: ", beacon_attention_mask[0][:10, :10])

# %%
def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            if use_custom_attn_mask:
                outputs = model(batch["input_ids"], labels=batch["input_ids"], attention_mask=beacon_attention_mask)
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
    project="beacon-attention",
    config={
        "use_custom_embedding": use_embedding,
        "use_custom_attn_mask": use_custom_attn_mask,
        "window_size": window_size,
        "use_normal_initialization": use_normal_initialization
    }
)

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

# %%
model_name = f"{'beacon_embed' if use_embedding else 'no_beacon_embed'}_{'beacon_attn_mask' if use_custom_attn_mask else 'regular_attn_mask'}_window_size_{window_size}_{'normal_init' if use_normal_initialization else 'zeros_init'}_model"
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

for epoch in range(num_epochs):
    for step, batch in tqdm(enumerate(train_dataloader, start=1), total=num_training_steps):
        T = batch['input_ids'].shape[1] # B, T
        if use_custom_attn_mask:
            logits = model(input_ids=batch["input_ids"], attention_mask=beacon_attention_mask).logits
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



