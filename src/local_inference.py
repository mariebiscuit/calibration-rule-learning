from typing import Tuple, List, Dict, Type
import json
import sys
import os
from tqdm import tqdm
import pandas as pd
import copy
import json
import torch
import re

from accelerate import init_empty_weights
from accelerate.utils.modeling import infer_auto_device_map, get_balanced_memory
from utils.llm_layers import *

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from peft import PeftModel
from datasets import Dataset

from utils.model_loading import load_model_and_tokenizer
from sparsification import load_wrapped_model
from losses import compute_distribution_loss

def _get_local_completions(model, tokenizer, dataset, anstoks):

    output = {k: [] for k in ['top5_ans', 'top_ans', 'sampled_ans', 'mass_ans', 'raw_true_mass', 'raw_false_mass', 'norm_true_mass']}

    dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=False, padding='max_length'), batched=True)

    with torch.no_grad():
        trues = torch.reshape(tokenizer(anstoks[0], return_tensors='pt', add_special_tokens=False)['input_ids'], (-1,))
        falses = torch.reshape(tokenizer(anstoks[1], return_tensors='pt', add_special_tokens=False)['input_ids'], (-1,))

        for i in range(len(dataset)):
            probs, _ = compute_distribution_loss(model, tokenizer, dataset[i], 0 , 1) 
            # dummy 0, 1 as t/f tokens since we don't need kl loss
            
            num_ans, vocab_sz = probs.shape
            top_ids = torch.argsort(probs, descending=True)[:, :5]
            top_answers = [[tokenizer.decode(t) for t in w] for w in top_ids]

            raw_true_mass = torch.sum(probs[:, trues], axis=-1).reshape((num_ans, -1))
            raw_false_mass = torch.sum(probs[:, falses], axis=-1).reshape((num_ans, -1))
            
            norm_true_mass = raw_true_mass / (raw_true_mass + raw_false_mass)
            norm_true_mass[torch.isnan(norm_true_mass)] = 0.5
            sampled_ans = torch.multinomial(torch.Tensor(torch.cat((1-norm_true_mass, norm_true_mass), axis=-1)), num_samples=1)

            output['top5_ans'] += top_answers
            output['top_ans'] += [top_ans[0] for top_ans in top_answers]
            output['mass_ans'] += torch.gt(raw_true_mass, raw_false_mass).reshape(-1).tolist()
            output['sampled_ans'].append(sampled_ans)
            output['raw_true_mass'].append(raw_true_mass)
            output['raw_false_mass'].append(raw_false_mass)
            output['norm_true_mass'].append(norm_true_mass)

        output['sampled_ans'] = torch.cat(output['sampled_ans'], dim=0).reshape(-1).tolist()
        output['raw_true_mass'] = torch.cat(output['raw_true_mass'], dim=0).reshape(-1).tolist()
        output['raw_false_mass']= torch.cat(output['raw_false_mass'], dim=0).reshape(-1).tolist()
        output['norm_true_mass'] = torch.cat(output['norm_true_mass'], dim=0).reshape(-1).tolist()

        return pd.DataFrame.from_dict(output)

def load_model(model_id, lora_checkpoint=None, circuit_model_checkpoint=None):

    if circuit_model_checkpoint is not None:
        model, tokenizer = load_wrapped_model(model_id, lora_checkpoint, 1e-8, 0, circuit_model_checkpoint)

    else:
        model, tokenizer = load_model_and_tokenizer(model_id, lora_checkpoint)
      
    return model, tokenizer

def query_local_completion(model_id: str,
            dataset: Type[Dataset], 
            uid: str, 
            load_in_8bit:bool = True,
            lora_checkpoint: str=None, 
            sparsification_checkpoint: str=None,
            anstoks=(["True", "true", " True", " true"], 
            ["False", "false", " False", " false"])):

    if not os.path.isdir(f'./results/raw_results/{uid}'):
        os.mkdir(f'./results/raw_results/{uid}')
    else:
        raise ValueError("UID already exists!")

    model, tokenizer = load_model(model_id, lora_checkpoint, sparsification_checkpoint)

    for concept in tqdm(dataset['concepts'], total=len(dataset['concepts'])):
        concept_dataset = dataset.filter(lambda x: x['concepts'] == concept)
        output_df = _get_local_completions(model, tokenizer, concept_dataset, anstoks=anstoks)
        output_df['concept'] = [concept] * len(output_df)
        output_df['uid'] = [uid] * len(output_df)
        output_df['answers'] = [a for sublist in concept_dataset['answers'][0] for a in sublist]
        output_df['object'] = [o for sublist in concept_dataset['objs'][0] for o in sublist]
    
        output_df.to_csv(f'./results/raw_results/{uid}/{uid}_{concept}.csv')

    print("Done evaluating!")

if __name__ == "__main__":

    import time
    import datasets
    from datasets.utils.logging import disable_progress_bar
    from utils.get_api_keys import HF_TOKEN
    disable_progress_bar()


    timestamp =  str(time.time()).split(".")[0][-5:]

    os.environ['HF_TOKEN'] = HF_TOKEN
    os.environ['TRANSFORMERS_CACHE'] = '/oscar/scratch/aloo1/model_cache_2'
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

    data = datasets.load_from_disk("./datasets/kl_dataset")["L2"]

    """
    Querying a pretrained model
    """
    # query_local_completion('google/gemma-2b', data, "gemma2b-pretrained")
    # query_local_completion('google/gemma-7b', data, "gemma7b-pretrained")

    """
    Querying a model with LoRA Checkpoint
    """ 
    # query_local_completion('google/gemma-2b', data, "gemma2b-tuned112",
    #                         lora_checkpoint="/users/aloo1/scratch/checkpoints_gemma_kl_fulldist_2b_5000/checkpoint-3000",
    #                         anstoks=([" True", " False"]))
    
    query_local_completion('google/gemma-7b', data, "gemma7b-pretrained",
                            # lora_checkpoint="/users/aloo1/scratch/checkpoints_gemma-7b-tuned92-and-special",
                            # lora_checkpoint="/users/aloo1/scratch/checkpoints_gemma-7b-tuned92",
                            # lora_checkpoint="/users/aloo1/scratch/checkpoints_gemma_kl_fulldist",
                            # lora_checkpoint='/users/aloo1/scratch/checkpoints_gemma-7b-tuned112-answerloss',
                            anstoks=([" True", " False"]))

    query_local_completion('google/gemma-7b', data, "gemma7b-tuned112-answerloss",
                            # lora_checkpoint="/users/aloo1/scratch/checkpoints_gemma-7b-tuned92-and-special",
                            # lora_checkpoint="/users/aloo1/scratch/checkpoints_gemma-7b-tuned92",
                            # lora_checkpoint="/users/aloo1/scratch/checkpoints_gemma_kl_fulldist",
                            lora_checkpoint='/users/aloo1/scratch/checkpoints_gemma-7b-tuned112-answerloss',
                            anstoks=([" True", " False"]))
    """
    Querying a model with LoRA Checkpoint + Sparsification
    """ 
    # query_local_completion('google/gemma-2b', data, "gemma2b-tuned112-sparsed_primitivesor-test",
    #                         lora_checkpoint="/users/aloo1/scratch/checkpoints_gemma-2b-tuned112/checkpoint-1500",
    #                         sparsification_checkpoint='/users/aloo1/scratch/checkpoints_gemma2b_100e_sparsing_manual_primitives_or/primitives_or_299_sparsified_checkpoint.pth',
    #                         anstoks=([" True", " False"]))

