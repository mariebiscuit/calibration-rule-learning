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

def _create_device_map(model_path: str) -> dict[str, int]:
    # credit to https://github.com/roeehendel/icl_task_vectors/blob/master/core/models/llm_loading.py
    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    layer_class = get_layers(model)[0].__class__.__name__

    max_memory = get_balanced_memory(model, no_split_module_classes=[layer_class])
    max_memory[0] = 0
    base_device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=[layer_class])

    num_devices = torch.cuda.device_count()

    if num_devices > 1:
        layers_path = get_layers_path(model)

        device_map_layers = {k: v for k, v in base_device_map.items() if k.startswith(layers_path)}
        device_map_other = {k: v for k, v in base_device_map.items() if k not in device_map_layers}

        # place the other layers on device 0
        device_map_other = {k: 0 for k in device_map_other}
        # split the layers evenly across the other devices (1-num_devices)
        num_layers = len(device_map_layers)
        num_layers_per_device = math.ceil(num_layers / (num_devices - 1))
        device_map_layers = {k: (i // num_layers_per_device + 1) for i, k in enumerate(device_map_layers)}

        device_map = {**device_map_other, **device_map_layers}
    else:
        device_map = {k: 0 for k, v in base_device_map.items()}
    return device_map

def _load_inference_model(model_id: str, lora_adaptor_fp: str = None):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    device_map = _create_device_map(model_id)

    base_model = AutoModelForCausalLM.from_pretrained(model_id, 
            quantization_config=bnb_config, 
            device_map=device_map,
            cache_dir=os.environ['TRANSFORMERS_CACHE'])

    if lora_adaptor_fp is not None:
        inf_model = PeftModel.from_pretrained(base_model, lora_adaptor_fp, 
            adapter_name="adapter1", 
            device_map='auto')
    else:
        inf_model = base_model
    
    for x in inf_model.parameters():
        x.requires_grad = False

    inf_model.eval()
    
    return inf_model

def query_local_completion(model_id: str,
            dataset: Type[Dataset], 
            uid: str, 
            load_in_8bit:bool = True,
            lora_checkpoint=None, 
            anstoks=(["True", "true", " True", " true"], 
            ["False", "false", " False", " false"])):

    if not os.path.isdir(f'./results/raw_results/{uid}'):
        os.mkdir(f'./results/raw_results/{uid}')

    device_map = _create_device_map(model_id)

    if lora_checkpoint is None:
        model = AutoModelForCausalLM.from_pretrained(model_id, 
            device_map=device_map, 
            load_in_8bit=load_in_8bit,
            cache_dir=os.environ['TRANSFORMERS_CACHE'])

        tokenizer = AutoTokenizer.from_pretrained(model_id, 
                    cache_dir=os.environ['TRANSFORMERS_CACHE'], 
                    add_eos_token=True)
        tokenizer.truncation_side='left'
        tokenizer.pad_token = tokenizer.eos_token

    else: 
        print("Unpacking LoRA checkpoint...")
        model = _load_inference_model(model_id, lora_checkpoint)

        tokenizer = AutoTokenizer.from_pretrained(model_id, 
                cache_dir=os.environ['TRANSFORMERS_CACHE'], 
                add_eos_token=True)

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
    
    query_local_completion('google/gemma-7b', data, "gemma7b-tuned112",
                            lora_checkpoint="/users/aloo1/scratch/checkpoints_gemma_kl_fulldist/checkpoint-2000",
                            anstoks=([" True", " False"]))
    

