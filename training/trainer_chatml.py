#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : trainer_chatml.py
@Author  : wenge-research
@Desc    : multiturn chat data sft using chatml template
'''

import os
import math
import pathlib, random
from typing import Optional, Dict
from dataclasses import dataclass, field
import json
import transformers
from transformers.training_args import TrainingArguments

import torch
from torch.utils.data import Dataset


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="wenge-research/yayi2-30b")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)


class SupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning.
    You can define your own chat template, Here is an example of ChatML:

    <|im_start|>system
    You are a helpful and harmless assistant named YAYI.<|im_end|>
    <|im_start|>user
    Hello!<|im_end|>
    <|im_start|>assistant
    Hello! How can I assist you today?<|im_end|>
    <|im_start|>user
    1+1=<|im_end|>
    <|im_start|>assistant
    1+1 equals 2.<|im_end|>
    
    """

    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length,
        system_token="<|im_start|>system\n",        # define your own role special token.
        user_token="<|im_start|>user\n",            
        assistant_token="<|im_start|>assistant\n",  
        eos_token="<|im_end|>\n",                   # define your own end token.
    ):
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length

        self.system_token_ids = self.tokenizer(system_token).input_ids
        self.user_token_ids = self.tokenizer(user_token).input_ids
        self.assistant_token_ids = self.tokenizer(assistant_token).input_ids
        self.eos_token_ids = self.tokenizer(eos_token).input_ids

        self.ignore_index = -100
        item = self.preprocessing(self.data[0])
        labels = []
        for id_ in item["labels"]:
            if id_ == -100:
                continue

            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))

    def __len__(self):
        return len(self.data)
        
    def preprocessing(self, example):
        
        input_ids = []
        labels = []

        # add system role prompt
        system = example.get("system", None)
        if not system:
            system = "You are a helpful and harmless assistant named YAYI."

        system_ids = self.tokenizer.encode(system)
        input_ids += self.system_token_ids + system_ids + self.eos_token_ids
        labels += [self.ignore_index] * len(self.system_token_ids + system_ids + self.eos_token_ids)

        # add human(user) / yayi(assistant) role prompt
        for message in example["conversations"]:
            from_ = message["from"]
            value = message["value"]
            value_ids = self.tokenizer.encode(value)

            if from_ in ["human", "user"]:
                input_ids += self.user_token_ids + value_ids + self.eos_token_ids
                labels += [self.ignore_index] * len(self.user_token_ids + value_ids + self.eos_token_ids)

            elif from_ in ["yayi", "assistant"]:
                input_ids += self.assistant_token_ids + value_ids + self.eos_token_ids
                labels += [self.ignore_index] * len(self.assistant_token_ids) + value_ids + self.eos_token_ids

        # padding or truncation
        input_ids = input_ids[:self.model_max_length]
        labels = labels[:self.model_max_length]
        
        assert len(input_ids)==len(labels)

        input_ids += [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids))
        labels += [self.ignore_index] * (self.model_max_length - len(labels))

        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    if training_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj"],
            inference_mode=False,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    dataset = SupervisedDataset(
        data_args.data_path, tokenizer, training_args.model_max_length
    )
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
