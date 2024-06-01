import pandas as pd
import numpy as np
from datasets import Dataset
import datasets

from accelerate import PartialState
import pickle

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
import transformers
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
from typing import Type, Dict, List

from utils.model_loading import _create_device_map
from losses import compute_distribution_loss, compute_restricted_ce_loss

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
    
def get_gemma_model(model_id, checkpoint:str=None, adapter_name="adapter1", ddp: bool=False):

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
            device_map={"": PartialState().process_index} if ddp else device_map
            )

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

class KLTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(KLTrainer, self).__init__(*args, **kwargs)
        self.true = self.tokenizer.encode(" True", add_special_tokens=False, return_tensors='pt').reshape(-1).to(torch.int64)
        self.false = self.tokenizer.encode(" False", add_special_tokens=False, return_tensors='pt').reshape(-1).to(torch.int64)

        with open('./datasets/kl_dataset/L1_hdists.pkl', 'rb') as f:
            self.train_hdists = pickle.load(f)
        
        with open('./datasets/kl_dataset/L2_hdists.pkl', 'rb') as f:
            self.test_hdists = pickle.load(f)

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs['hdist'] = [self.train_hdists[i] for i in inputs['idx']]
        _, kl = compute_distribution_loss(model, self.tokenizer, inputs, self.true, self.false)
        return kl
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        if self.eval_dataset is not None:
            with torch.inference_mode():
                test_losses = []
                test_coef = []
                for i in range(len(self.eval_dataset)):
                    inputs = self.eval_dataset[i]

                    if isinstance(inputs['idx'], int):
                        inputs['hdist'] = self.test_hdists[inputs['idx']]
                    else:
                        inputs['hdist'] = [self.test_hdists[i] for i in inputs['idx']]
                    
                    probs, kl_loss = compute_distribution_loss(self.model, self.tokenizer, 
                                inputs, self.true, self.false)
                    
                    hdist = self.test_hdists[self.eval_dataset[i]['idx']][:, 0]
                    mdist = probs[:, self.true].reshape(-1)

                    r2 = np.corrcoef(mdist, hdist)[0, 1] ** 2

                    test_losses.append(kl_loss)
                    test_coef.append(r2)

                kl_loss = np.mean(test_losses)
                r2 = np.mean(np.mean(test_coef))
                output = {"KL": kl_loss, "R2": r2}
                self.log(output)

            return output
        else:
            return None


class AnswerCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        _, ce = compute_restricted_ce_loss(model, self.tokenizer, inputs)

        return ce
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        return None # TODO: Not supported

def train(model_factory: callable, 
          tokenizer_factory: callable, 
          train_dataset: Type[Dataset],
          eval_dataset: Type[Dataset],
          run_name:str,
          output_dir:str,
          distribution_loss=True,
          ddp:bool = False,
          load_checkpoint:str=None,
          num_train_epochs:int=None,
          max_steps:int=1000,
          save_steps:int=200,
          fp16:bool=True,
          optim:str="paged_adamw_8bit",
          gradient_checkpointing:bool=True,
          gradient_checkpointing_kwargs={"use_reentrant":False},
          gradient_accumulation_steps:int=4,
          per_device_train_batch_size:int=1):
    """
    Performs training with HF Trainer
    
    With reference from:
       - https://medium.com/@wwang1110/qlora-and-hf-trainer-with-custom-model-da636b0f7389
       - https://medium.com/@samvardhan777/fine-tune-gemma-using-qlora-%EF%B8%8F-6b2f2e76dc55
       - https://stackoverflow.com/questions/75814047/how-to-use-huggingface-trainer-with-multiple-gpus
    """

    transformers.logging.set_verbosity_info()

    tokenizer = tokenizer_factory()
    
    tokenizer.truncation_side='left'
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = train_dataset.map(lambda e: tokenizer(e['text'], truncation=False, padding='max_length'), batched=True)
    print("dataset length", len(train_dataset['text']))
    print(train_dataset['text'][2])

    train_dataset = train_dataset.select_columns(['input_ids', 'attention_mask', 'idx'])

    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(lambda e: tokenizer(e['text'], truncation=False, padding='max_length'), batched=True)
        eval_dataset = eval_dataset.select_columns(['input_ids', 'attention_mask', 'idx'])

    model= model_factory()

    trainer_arguments = dict(model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=TrainingArguments(
                report_to="wandb",
                run_name=run_name,
                remove_unused_columns=False,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_checkpointing=gradient_checkpointing,
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=30,
                max_steps=max_steps,
                num_train_epochs=num_train_epochs,
                logging_steps=5,
                learning_rate=1e-5,
                fp16=fp16,
                evaluation_strategy='steps',
                eval_steps=50,
                save_strategy='steps',
                save_steps=save_steps,
                output_dir=output_dir,
                optim=optim,
            ))

    torch.cuda.empty_cache()
    if distribution_loss:
        trainer = KLTrainer(**trainer_arguments)
    else:
        trainer = AnswerCETrainer(**trainer_arguments)

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    print("Done training!")

def main():

    """
    Example call:
    `python3 ./src/lora_tuning.pys run_name tuned112 google/gemma-7b 2000`
    python3 -m torch.distributed.launch --nproc_per_node 2 ./src/lora_tuning.py gemma-2b-tuned112 tuned112 google/gemma-2b
    """
    print(sys.argv[0])
    RUN_NAME = sys.argv[-3]
    SUBSET_NAME = sys.argv[-2]
    MODEL_ID = sys.argv[-1]

    from utils.get_api_keys import HF_TOKEN
    from utils.get_concept_subsets import SUBSETS

    print(f"Starting run {RUN_NAME} on subset {SUBSET_NAME}...")

    torch.cuda.empty_cache()
    os.environ["WANDB_PROJECT"]=f"{RUN_NAME}"
    os.environ['HF_TOKEN'] = HF_TOKEN
    os.environ['TRANSFORMERS_CACHE'] = '/oscar/scratch/aloo1/model_cache_2'

    concept_list = SUBSETS[SUBSET_NAME]
    data = datasets.load_from_disk("./datasets/kl_dataset")
    train_data = data["L1"].filter(lambda x: x['concepts'] in concept_list)
    # test_data = data["L2"].filter(lambda x: x['concepts'] in concept_list)

    DDP=True
    train(model_factory = lambda: get_gemma_model(MODEL_ID, ddp=DDP),
          tokenizer_factory=lambda: get_tokenizer(MODEL_ID),
          train_dataset=train_data, 
          eval_dataset=None, 
          run_name=RUN_NAME, 
          distribution_loss=False,
          output_dir= f'/users/aloo1/scratch/checkpoints_{RUN_NAME}',
          max_steps=0, 
          num_train_epochs=72,
          save_steps=500,
          gradient_accumulation_steps=4)


if __name__ == "__main__":
    main()