import pandas as pd
import numpy as np
from datasets import Dataset
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
import transformers
import sys
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
from typing import Type, Dict, List

sys.path.append('/users/aloo1/thesis/task_vector_exp/icl_task_vectors')
from core.models.llm_loading import _create_device_map

sys.path.append('/users/aloo1/thesis')
from utils.training import KLTrainer, RestrictedBackpropTrainer, NormalTrainer, get_target_idxes

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
    return list(lora_module_names)
    
def get_gemma_model(model_id, checkpoint:str=None, adapter_name="adapter1"):

    device_map = _create_device_map(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, 
            quantization_config=bnb_config, 
            cache_dir=os.environ['TRANSFORMERS_CACHE'],
            device_map=device_map)

    modules = find_all_linear_names(model)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=modules,
        use_rslora=True,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    if checkpoint is None:
        model = get_peft_model(model, lora_config)
    else:
        print("Using checkpoint ", checkpoint)
        model = PeftModel.from_pretrained(model, checkpoint,
                adapter_name=adapter_name,
                device_map='auto', is_trainable=True)
    
    model.print_trainable_parameters()
    return model

def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, 
                cache_dir=os.environ['TRANSFORMERS_CACHE'], 
                add_eos_token=True)
    return tokenizer

def train(model_factory, 
          tokenizer_factory, 
          train_dataset: Type[Dataset],
          loss_type: str,
          run_name=str,
          load_checkpoint:str=None,
          max_steps:int=1000,
          save_steps:int=200,
          fp16:bool=True,
          optim:str="paged_adamw_8bit",
          gradient_checkpointing:bool=True,
          gradient_checkpointing_kwargs={"use_reentrant":False},
          per_device_train_batch_size:int=1):

    assert loss_type in ["KL", "RESTRICTED_LM", "LM"], 'Please enter a loss_type parameter within ["KL", "RESTRICTED_LM", "LM"]'

    transformers.logging.set_verbosity_info()
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

    tokenizer = tokenizer_factory()
    
    tokenizer.truncation_side='left'
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = train_dataset.map(lambda e: tokenizer(e['text'], truncation=False, padding='max_length'), batched=True)
    print("dataset length", len(train_dataset['text']))
    print(train_dataset['text'][2])

    if loss_type == "KL": 
        train_dataset = train_dataset.select_columns(['input_ids', 'attention_mask', 'idx']) # need idx column to access right hdists
    elif loss_type in ["RESTRICTED_LM" or "LM"]:
        train_dataset = train_dataset.select_columns(['input_ids', 'attention_mask'])
    else:
        raise ValueError("Invalid loss type") #shouldn't get here

    model= model_factory()
    output_dir = f'/users/aloo1/scratch/checkpoints_{run_name}'

    trainer_arguments = dict(model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            args=TrainingArguments(
                prediction_loss_only=True,
                report_to="wandb",
                run_name=run_name,
                remove_unused_columns=False,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_checkpointing=gradient_checkpointing,
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
                gradient_accumulation_steps=4,
                warmup_steps=30,
                max_steps=max_steps,
                logging_steps=5,
                learning_rate=1e-5,
                fp16=fp16,
                save_strategy='steps',
                save_steps=save_steps,
                output_dir=output_dir,
                optim=optim,
            ))

    torch.cuda.empty_cache()

    if loss_type == "KL":
        trainer = KLTrainer(**trainer_arguments)
    elif loss_type == "RESTRICTED_LM":
        trainer = RestrictedBackpropTrainer(**trainer_arguments)
    elif loss_type == "LM":
        trainer = Trainer(**trainer_arguments)
    else:
        raise ValueError("Invalid loss type") #shouldn't get here
    
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    print("Done training!")

    # https://medium.com/@wwang1110/qlora-and-hf-trainer-with-custom-model-da636b0f7389
    # https://medium.com/@samvardhan777/fine-tune-gemma-using-qlora-%EF%B8%8F-6b2f2e76dc55
    # https://stackoverflow.com/questions/75814047/how-to-use-huggingface-trainer-with-multiple-gpus

if __name__ == "__main__":

    RUN_NAME = sys.argv[1]
    print(f"Starting run {RUN_NAME}...")

    datasets = {'kl': '/users/aloo1/thesis/rq2_fit/train_kl_label_dataset.csv', 
               'indiv': '/users/aloo1/thesis/rq2_fit/indiv_dataset_train.csv',
               'agg': '/users/aloo1/thesis/rq2_fit/train_agg_dataset.csv'}

    with open('/users/aloo1/thesis/rq2_fit/held-out-concepts.txt', 'r') as f:
        heldout = [x.strip() for x in f.readlines()]
    
    with open('/users/aloo1/thesis/rq2_fit/primitives.txt', 'r') as f:
        primitives = [x.strip() for x in f.readlines()]

    with open('/users/aloo1/thesis/sfl-data/boolean_concepts.txt', 'r') as f:
        boolean = [x.strip() for x in f.readlines()]

    torch.cuda.empty_cache()
    os.environ["WANDB_PROJECT"]=f"{RUN_NAME}"
    os.environ['HF_TOKEN'] = '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'
    os.environ['TRANSFORMERS_CACHE'] = '/oscar/scratch/aloo1/model_cache_2'


    datasets = Dataset.from_csv(datasets['kl'])
    # datasets = datasets.filter(lambda x: x['concepts'] in primitives)
    train(model_factory = lambda: get_gemma_model('google/gemma-2b'), tokenizer_factory=lambda: get_tokenizer('google/gemma-2b'),
         train_dataset=datasets, loss_type='KL', run_name=RUN_NAME, max_steps=5000, save_steps=500)
