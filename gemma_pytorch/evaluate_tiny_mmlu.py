# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import contextlib
import random

import numpy as np
import torch

from gemma_pytorch.gemma import config
from gemma_pytorch.gemma import model as gemma_model
from datasets import load_dataset


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

def main(args):
    # Construct the model config.
    model_config = config.get_model_config(args.variant)
    model_config.dtype = "float32" if args.device == "cpu" else "float16"
    model_config.quant = True

    # Seed random.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create the model and load the weights.
    device = torch.device(args.device)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma_model.GemmaForCausalLM(model_config)
        model.load_weights(args.ckpt)
        model = model.to(device).eval()
    print("Model loading done")

    dataset = load_dataset("tinybenchmarks/tinyMMLU")
    for batch in dataset.iter(batch_size=args.batch_size):
        formatted_input = batch["input_formatted"]
        result = model.generate(formatted_input, device, output_len=args.output_len, token_compression=args.token_compression)
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--variant",
                        type=str,
                        default="2b",
                        choices=["2b", "7b"])
    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--output_len", type=int, default=100)
    parser.add_argument("--token_compression", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--quant", action='store_true')
    args = parser.parse_args()

    main(args)
