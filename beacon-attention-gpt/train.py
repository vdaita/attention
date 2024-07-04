from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from openai import OpenAI
from dotenv import load_dotenv
from utils import BeaconEmbedding, generate_beacon_attention_mask_2d
import os
import torch
from torch.optim import AdamW
from transformers import DataCollatorForLanguageModeling
import time

load_dotenv(".env")

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_model(use_embedding: bool, use_custom_attn_mask: bool, num_epochs: int = 1, output_dir: str = "./outputs/"):
    model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    model.to(device)

    if use_embedding:
        beacon_embedding = BeaconEmbedding(vocab_size=model.config.vocab_size, n_embed=model.config.n_embd, window_length=4)
        model.set_input_embeddings(beacon_embedding)

    model.train()
    attention_mask = generate_beacon_attention_mask_2d(256, device)
    
    step_times = []

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = num_epochs * len(train_dataloader)

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            start_time = time.perf_counter()

            batch = {k: v.to(device) for k, v in batch.items()}
            # Full attention mask makes sure that padding is also fully masked. 
            T = batch["input_ids"].shape[1] # shape[0] should be number of batches
            
            batch_attn_mask = batch["attention_mask"]
            if use_custom_attn_mask:
                batch_attn_mask = batch_attn_mask & attention_mask[:T, :T]

            outputs = model(input_ids=batch["input_ids"], attention_mask=batch_attn_mask, position_ids=batch["position_ids"])
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            end_time = time.perf_counter()
            step_times.append(end_time - start_time)

    model_name = f"{'beacon_embed' if use_embedding else 'no_beacon_embed'}_{'beacon_attn_mask' if use_custom_attn_mask else 'regular_attn_mask'}_model"
    if not(os.path.exists(output_dir)):
        os.makedirs(output_dir)
    model.save_pretrained(f"{output_dir}/{model_name}", from_pt=True)

    print(f"Output folder: {output_dir}/{model_name}")
    print(f"Mean training step time: ", sum(step_times)/len(step_times))