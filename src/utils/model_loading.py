
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


def _create_device_map(model_path: str) -> dict[str, int]:
    """
    All credit to https://github.com/roeehendel/icl_task_vectors/blob/master/core/models/llm_loading.py
    """
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


def load_model_and_tokenizer(model_id: str, lora_checkpoint: str =None):

    model = _load_inference_model(model_id, lora_checkpoint)
                                    
    tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=os.environ['TRANSFORMERS_CACHE'], 
            add_eos_token=True)

    tokenizer.truncation_side='left'
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
