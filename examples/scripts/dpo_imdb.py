# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""
# regular:
python examples/scripts/dpo.py \
    --dataset_name=trl-internal-testing/hh-rlhf-helpful-base-trl-style \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="dpo_anthropic_hh" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

# peft:
python examples/scripts/dpo.py \
    --dataset_name=trl-internal-testing/hh-rlhf-helpful-base-trl-style \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="dpo_anthropic_hh" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""

import logging
import multiprocessing
import os
import re
from pathlib import Path
from contextlib import nullcontext

from trl.commands.cli_utils import DPOScriptArguments, init_zero_verbose, TrlParser
from trl.env_utils import strtobool

TRL_USE_RICH = strtobool(os.getenv("TRL_USE_RICH", "0"))

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

def load_imdb_data(directory: Path):
    # for train_test in ['train', 'test']:    
    data = {'rejected': [], 'chosen': [], 'prompt': []}
    for label in ['neg', 'pos']:
        # subdir = os.path.join(directory, label)
        subdir = directory / label
        for filename in subdir.iterdir():
            if filename.is_file() and filename.suffix == '.txt':
                with open(filename, 'r', encoding='utf-8') as file:
                    text = file.read().strip()
                match = re.match(r'(\d+)_(\d+)\.txt', filename.name)
                if match:
                    id_, rating = match.groups()
                    # data['text'].append(text)
                    # data['rating'].append(int(rating))
                    if label == 'neg':
                        data['prompt'].append(text)
                        data['chosen'].append(f'This is a negative review with a rating of {rating}.')
                        data['rejected'].append(f'This is a positive review with a rating of {rating}.')
                    else:
                        data['prompt'].append(text)
                        data['chosen'].append(f'This is a positive review with a rating of {rating}.')
                        data['rejected'].append(f'This is a negative review with a rating of {rating}.')
        # res[train_test] = data
    
    return Dataset.from_dict(data)


if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )
    else:
        model_ref = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the DPOTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Dataset
    ################
    
    
    # ds = load_dataset(args.dataset_name)
    train_ds = load_imdb_data(Path('/home/jovyan/notebook/trl/datasets/aclImdb/train'))
    eval_ds = load_imdb_data(Path('/home/jovyan/notebook/trl/datasets/aclImdb/test'))
    # ds = load_dataset("imdb", split="train")
    
    # if args.sanity_check:
    #     for key in ds:
    #         ds[key] = ds[key].select(range(50))
    # breakpoint()
    # def build_prompt(example):
    #     prompt = f"Movie: {example['text']}"
    #     return prompt
    # breakpoint()
    # ds = ds.map(build_prompt, batched=True, desc="Building prompts")
    
    # def build_responses(example):
    #     if example["label"] >= 7:  # positive review
    #         example['prompt'] = f"Movie: {example['text']}"
    #         example["chosen"] = example["text"]
    #         example["rejected"] = "This movie is terrible, I didn't like it at all."
    #     else:  # negative review
    #         example["chosen"] = "This is one of the best movies I've ever seen!"
    #         example["rejected"] = example["text"]
    #     return example

    # ds = ds.map(build_responses)
    
    # breakpoint()
    
    def process(row):
        row["prompt"] = tokenizer.apply_chat_template([{'content': row["prompt"], 'role':'user'}], tokenize=False)
        row["chosen"] = tokenizer.apply_chat_template([{'content': row["chosen"], 'role': 'assistant'}], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template([{'content': row["rejected"], 'role': 'assistant'}], tokenize=False)
        return row
    train_ds = train_ds.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    
    eval_ds = eval_ds.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    
    # dataset.set_format(type="keras", columns=["prompt", "chosen", "rejected"])
    # train_dataset = ds[args.dataset_train_split]
    # eval_dataset = ds[args.dataset_test_split]
    # breakpoint()

    ################
    # Training
    ################
    with init_context:
        trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            peft_config=peft_config,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
