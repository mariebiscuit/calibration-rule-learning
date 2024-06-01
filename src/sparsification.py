import os

import pandas as pd
import numpy as np
import pickle 

from typing import Dict, Union, Any, Tuple
from tqdm import tqdm

from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
import datasets
import torch

from torch import nn
from torch.utils.data import DataLoader
from NeuroSurgeon.Models import model_configs, circuit_model

from losses import compute_distribution_loss
from utils.model_loading import load_model_and_tokenizer

class TemperatureCallback:
    # A simple callback that updates the probes temperature parameter,
    # which transforms a soft mask into a hard mask
    def __init__(self, total_epochs, final_temp):
        self.temp_increase = final_temp ** (1.0 / total_epochs)

    def update(self, model):
        temp = model.temperature
        model.temperature = temp * self.temp_increase


class SparsificationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SparsificationTrainer, self).__init__(*args, **kwargs)
        self.true = self.tokenizer.encode(" True", add_special_tokens=False, return_tensors='pt').reshape(-1).to(torch.int64)
        self.false = self.tokenizer.encode(" False", add_special_tokens=False, return_tensors='pt').reshape(-1).to(torch.int64)
        self.temp_callback = TemperatureCallback(self.args.max_steps // len(self.train_dataset), 150)
        self.steps = 0

        with open('./datasets/kl_dataset/L1_hdists.pkl', 'rb') as f:
            self.train_hdists = pickle.load(f)
        
        with open('./datasets/kl_dataset/L2_hdists.pkl', 'rb') as f:
            self.test_hdists = pickle.load(f)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        """
        # inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)
        self.temp_callback.update(self.model)
        self.steps += 1

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs['hdist'] = [self.train_hdists[i] for i in inputs['idx']]
        _, kl = compute_distribution_loss(model, self.tokenizer, inputs, self.true, self.false)
        
        loss = kl + (self.model.config.l0_lambda * self.model._compute_l0_loss())
        return loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
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
            L0 = self.model._compute_l0_loss()
        
        return {"KL": kl_loss, "R2": r2, "L0": L0}

def load_wrapped_model(model_id: str, 
                    load_checkpoint: str, 
                    l0_lambda: float, 
                    start_layer: int, 
                    load_sparsification_checkpoint: str = None):

    model, tokenizer = load_model_and_tokenizer(model_id, load_checkpoint)

    target_layers = list(model.state_dict().keys())
    target_layers = [
        ".".join(target_layer.split(".")[:-1])
        for target_layer in target_layers
        if (
        (("layers" in target_layer) 
            and (target_layer.endswith('weight')) 
            and ("layernorm" not in target_layer)
            and (int(target_layer.split(".")[4]) >= start_layer)
            and ("mlp" in target_layer)
            and ("down_proj" in target_layer)
            and ("lora_A" not in target_layer)
            ) 
        )
    ]

    config = model_configs.CircuitConfig(
        mask_method="continuous_sparsification", # Binary Mask Optimization method
        mask_hparams = {
            "ablation": "none", # Don't invert the learned mask
            "mask_unit": "weight", # Mask at the weight-level
            "mask_bias": False, # Don't mask biases
            "mask_init_value": 0.0 # Initialize the mask parameters at 0
        },
        target_layers=target_layers, # Replace the layers specified above
        freeze_base=True, # Don't train the model weights when training the mask
        add_l0=False, # Use L0 Regularization
        l0_lambda=l0_lambda, # Multiplier on L0 norm for balancing the loss function
    )

    model = circuit_model.CircuitModel(config, model)

    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)
    
    if load_sparsification_checkpoint is not None:
        checkpoint = torch.load(load_sparsification_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        stats = model.compute_l0_statistics()
        print("Sparsification checkpoint \%", stats['total_l0'].item() / stats['max_l0'])
        print(stats)

    return model, tokenizer

# this gives OOM because loss needs to be backwarded and clearaed for each input
def run_hf_trainer(conditions: Dict,
          run_name:str,
          load_checkpoint:str,
          output_dir: str,
          start_layer:int=0,
          l0_lambda:float = 1e-8,
          max_steps:int=1000,
          save_steps:int=100,
          fp16:bool=True,
          optim:str="paged_adamw_8bit",
          gradient_checkpointing:bool=False,
          gradient_checkpointing_kwargs=None,
          per_device_train_batch_size:int=1):

    for condition, concept_list in conditions.items():
        torch.cuda.empty_cache()
        model, tokenizer = load_wrapped_model(load_checkpoint, l0_lambda, start_layer)

        data = datasets.load_from_disk("./datasets/kl_dataset")
        train_dataset = data["L1"].filter(lambda x: x['concepts'] in concept_list).map(
            lambda e: tokenizer(e['text'], truncation=False, padding='max_length'), batched=True)
        test_dataset = data["L2"].filter(lambda x: x['concepts'] in concept_list).map(
            lambda e: tokenizer(e['text'], truncation=False, padding='max_length'), batched=True)

        train_dataset = train_dataset.select_columns(['input_ids', 'attention_mask', 'idx'])
        test_dataset = test_dataset.select_columns(['input_ids', 'attention_mask', 'idx'])

        trainer_arguments = dict(model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            args=TrainingArguments(
                prediction_loss_only=True,
                report_to="wandb",
                run_name=run_name,
                remove_unused_columns=False,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_checkpointing=gradient_checkpointing,
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
                gradient_accumulation_steps=2,
                warmup_steps=30,
                max_steps=max_steps,
                logging_steps=5,
                learning_rate=1e-5,
                fp16=fp16,
                save_strategy='steps',
                save_steps=save_steps,
                evaluation_strategy='steps',
                eval_steps=20,
                output_dir=output_dir,
                optim=optim,
            ))
        
        torch.cuda.empty_cache()
        trainer = SparsificationTrainer(**trainer_arguments)
        trainer.train()
        trainer.model.save_pretrained(output_dir)
        print("Done training!")


def run_manual_trainer( conditions: dict,
                        run_name: str,
                        load_checkpoint: str,
                        output_dir: str,
                        start_layer: int=8,
                        load_prev_epoch: Tuple[str, int]= None,
                        l0_lambda: float=1e-8,
                        num_epochs=300,
                        batch_size=2, 
                        final_temp=150,
                        eval_every: int = 10,
                        checkpoint_every: int = 20):

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    model, tokenizer = load_wrapped_model("google/gemma-2b", load_checkpoint, l0_lambda, start_layer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    start_epoch = 0

    if load_prev_epoch is not None:
        state_dict, start_epoch = load_prev_epoch
        print(f"Loading checkpoint from {state_dict}...")
        checkpoint = torch.load(state_dict)
        model.load_state_dict(checkpoint['state_dict'])

    for condition, concept_list in conditions.items():
        data = datasets.load_from_disk("./datasets/kl_dataset")
        train_data = data["L1"].filter(lambda x: x['concepts'] in concept_list).map(
                lambda e: tokenizer(e['text'], truncation=False, padding='max_length'), batched=True)
        test_data = data["L2"].filter(lambda x: x['concepts'] in concept_list).map(
                lambda e: tokenizer(e['text'], truncation=False, padding='max_length'), batched=True)

        train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=lambda x: x)
        test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=lambda x: x)

        temp_callback = TemperatureCallback(num_epochs, final_temp)
        truetok = tokenizer.encode(" True", add_special_tokens=False, return_tensors='pt').reshape(-1).to(torch.int64)
        falsetok = tokenizer.encode(" False", add_special_tokens=False, return_tensors='pt').reshape(-1).to(torch.int64)

        torch.cuda.empty_cache()
        progress_bar = tqdm(range(start_epoch, num_epochs))

        for epoch in range(start_epoch, num_epochs):
            total_train_loss = torch.zeros((1)).to(model.wrapped_model.device)
            concept_order = []
            for batch in train_dataloader:
                concept_order.append(batch[0]['concepts'])
                train_logits, train_loss = compute_distribution_loss(model, tokenizer, batch[0], truetok, falsetok)
                train_loss += (model.config.l0_lambda * model._compute_l0_loss()).to(train_loss.device) # Manually adding L0 Loss

                total_train_loss += train_loss
                train_loss.backward() # have to backprop every concept or else OOM
                optimizer.step()
                optimizer.zero_grad()
                    
            progress_bar.update(1)
            temp_callback.update(model)

            if epoch % eval_every == 0:
                curr_test_loss = []
                with torch.inference_mode():
                    for batch in test_dataloader:
                        _, test_loss =  compute_distribution_loss(model, tokenizer, batch[0], truetok, falsetok)
                        curr_test_loss.append(test_loss.cpu())

                    progress_bar.set_description(f"Epoch {epoch}: Test Loss {str(np.mean(curr_test_loss))[:7]} | " +\
                        f"L0 {model._compute_l0_loss()} | Train Loss {total_train_loss.item() / len(train_data)}")
        
            if epoch % checkpoint_every == 0:
                checkpoint = {'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}
                torch.save(checkpoint, os.path.join(output_dir, f"{condition}_{epoch}_sparsified_checkpoint.pth"))

        checkpoint = {'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}
        torch.save(checkpoint, os.path.join(output_dir, f"{condition}_{epoch}_sparsified_checkpoint.pth"))


if __name__ == "__main__":

    from utils.get_api_keys import HF_TOKEN

    RUN_NAME= "gemma2b_100e_sparsing_manual_primitives_or"

    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    os.environ["WANDB_PROJECT"]=f"{RUN_NAME}"
    os.environ['TRANSFORMERS_CACHE'] = '/oscar/scratch/aloo1/model_cache_2'
    os.environ['HF_TOKEN'] = HF_TOKEN

    conditions = {
        # "primitives_only": ['hg03', 'hg04', 'hg18', 'hg19', 'hg20'],
        "primitives_or": ['hg03', 'hg04', 'hg18', 'hg19', 'hg20', 'hg06', 'hg25'],
        # "primitives_and": ['hg03', 'hg04', 'hg18', 'hg19', 'hg20', 'hg24', 'hg09']
    }

    run_manual_trainer(conditions=conditions, run_name=RUN_NAME, 
                load_checkpoint="/users/aloo1/scratch/checkpoints_gemma-2b-tuned112/checkpoint-1500",
                # load_prev_epoch=("/users/aloo1/scratch/checkpoints_gemma2b_sparsing_manual_primitives_or/primitives_or_88_sparsified_checkpoint.pth", 89),
                output_dir = f"/users/aloo1/scratch/checkpoints_{RUN_NAME}",
                start_layer=0, eval_every=10, checkpoint_every=20)
