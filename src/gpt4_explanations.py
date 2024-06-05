import sys, os
import pandas as pd
import numpy as np
from typing import Type, Tuple, List, Dict
from collections import defaultdict
import time
from tqdm import tqdm
import re
import json

from utils.custom_data import get_perm, make_all_objects, paragraph_rep, make_rule
from utils.custom_data import FeatureSet, Obj, animal_features
from utils.api_utils import get_together_completion, get_chatcompletion
from utils.get_concept_subsets import SUBSETS

def parse_to_object(string: str) -> Type[Obj]:
    """
    Input: "triangle,yellow,2"
    Output: Obj object
    """
    shape, color, size = string.split(",")
    size = ["small", 'medium', 'large'][int(size) - 1]
    return Obj(**dict(size=size, color=color, shape=shape))
    

def extract_lists(concepts: List[str], listnums: List[str] = ["L2"], flatten=True) -> Dict[str, Tuple[List[Obj], List[bool]]]:
    """
    Given a list of desired concepts, returns subset of labels_to_data dictionary
    only with those concepts, up to 25 sets, only from the specified lists
    """
    data = {}
    with open('./data/labels_to_data.json', 'r') as f:
        datadict = json.load(f)
    
    for c in concepts:
        for listnum in listnums:
          if flatten:
            objects = [parse_to_object(x) for i, l in enumerate(datadict[c][listnum]['sets']) for x in l if i < 25]
            labels = [bool(x) for i, l in enumerate(datadict[c][listnum]['answers']) for x in l if i < 25]
          else:
            objects = [[parse_to_object(x) for x in l if i < 25] for i, l in enumerate(datadict[c][listnum]['sets'])]
            labels = [[bool(x)  for x in l if i < 25] for i, l in enumerate(datadict[c][listnum]['answers'])]

          data[c+"-"+listnum] = (objects, labels)

    return data

def make_system_message(message: str):
  return [{"role": 'system', "content": message}]

def make_experiment_block_from_list(
    n_examples: int, 
    rule: str,
    labels: str,
    objects: List[Obj],
    list_label: int,
    prompt: str,
    bookend: str,
    debrief: str=None):

    messages = []

    objects = objects[:n_examples]

    examples, answer = paragraph_rep(num=len(objects), 
                objects=[objects], 
                labels=[labels],
                show_last=False)
    
    query_example = examples.split("\n")[-1][:-1]

    messages.append(
      {"role": "user", "content": prompt + examples + bookend}
    )

    if debrief is not None:
      debrief = f"Based on your examples, I've learned that {debrief}{query_example}: {answer}"
      messages.append(
        {"role": "assistant", "content": debrief}
      )

    info = dict(
      num_examples = n_examples,
      rule = rule,
      list_label=list_label,
      examples=examples, 
      answer=answer, 
      query=query_example)
      
    return messages, info



def make_experiment_block(
    n_examples: int, 
    rule_name: str,
    feature_set: Type[FeatureSet], 
    perm_num: int,
    prompt: str, 
    bookend: str,
    debrief: str=None):

    messages = []
    permutation = get_perm(perm_num)
    objects = make_all_objects(feature_set)
    objects = objects[permutation][:n_examples]

    rule, _ = make_rule(rule_name, feature_set)

    examples, answer = paragraph_rep(num=len(objects), 
                objects=[objects], 
                labels=[[rule(ob) for ob in objects]],
                show_last=False)
    
    query_example = examples.split("\n")[-1][:-1]

    messages.append(
      {"role": "user", "content": prompt + examples + bookend}
    )

    if debrief is not None:
      debrief = f"Based on your examples, I've learned that {debrief}{query_example}: {answer}"
    
      messages.append(
        {"role": "assistant", "content": debrief}
      )

    info = dict(
      num_examples = n_examples,
      permutation=perm_num,
      rule= rule_name,
      examples=examples, 
      answer=answer, 
      query=query_example)
      
    return messages, info

def test_blockmaker():
    # == PRACTICE BLOCK ==
    practice_prompt = "I have Rule 0 in mind. I have labelled animals that follow Rule 0 True,"+\
                        "and animals that do not follow Rule 1 False.\n" +\
                        "Here are examples of labels based on Rule 1.\n"
                
    practice_bookend = "\nWhat's the label of the last object based on Rule 0?"
    practice_rule = "is_striped"
    practice_debrief = "Rule 0 is that animals must be striped.\n" +\
                        "Therefore, the label for the last object is\n"

    practice_block, info = make_experiment_block(
        n_examples = 15, 
        rule_name = practice_rule,
        feature_set = animal_features, 
        permutation=get_perm(400), 
        prompt=practice_prompt, 
        bookend=practice_bookend,
        debrief=practice_debrief)

    print("\n[Block]\n" + str(practice_block))
    print("\n[Info]\n" + str(info))

def construct_pre_prime(
    feature_set: Type[FeatureSet]=animal_features, 
    practice_perm: int=100,
    n_examples=15):

  pre_prime_messages = []

  # == SYSTEM_BLOCK ==
  system_msg =  "Users will show you examples that are labeled based on a rule. Your task is to " +\
                "deduce their rule from the examples and label the last example correctly. " +\
                "Make sure your rule correctly accounts for how all the objects so far are labeled." +\
                "Begin your response with 'Based on your examples, I've learned that...' and " +\
                "give your best concise guess of the rule without asking for more evidence. " +\
                "You absolutely must give a True or False label, even if you are not sure. " +\
                "End your reply with the last object and its label based on the rule."

  pre_prime_messages += make_system_message(system_msg)

  # == PRACTICE BLOCK ==
  practice_prompt = "I have Rule 0 in mind. I have labeled animals that follow Rule 0 True,"+\
                    "and animals that do not follow Rule 0 False.\n" +\
                    "Here are examples of labels based on Rule 0.\n"
              
  practice_bookend = "\nWhat's the label of the last animal based on Rule 0?"
  practice_rule = "is_striped"
  practice_debrief = "Rule 0 is that animals must be striped.\nTherefore, the label for the last animal is\n"

  practice_block, _ = make_experiment_block(
    n_examples = n_examples, 
    rule_name = practice_rule,
    feature_set = feature_set, 
    perm_num=practice_perm, 
    prompt=practice_prompt, 
    bookend=practice_bookend,
    debrief=practice_debrief)

  pre_prime_messages += practice_block
  return pre_prime_messages

def build_target_only_df(target_infos: List[List[Dict]]):
    output_df = defaultdict(list)
    for run_info in target_infos:
        for target_info in run_info:
            for key, item in target_info.items():
                if item is not None:
                    output_df['target_' + key].append(item)

    return pd.DataFrame.from_dict(output_df)

def extract_answer(string):
    regex = r': (True|False)'
    answer = re.search(regex, string)
    if answer:
        return answer.group(1)
    else:
        return "NA"


def run_lists(uid:str, concepts: List[str], listnums: List[str]=["L2"], mock: bool=False,
                  log:bool=True,
                  practice_perm=100):
    
    output_dir = f"./results/experiment_2/raw_results/{uid}"
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)
      os.mkdir(os.path.join(output_dir, "logs"))
    else:
      raise ValueError("Folder with UID already exists!")

    target_infos = []

    if mock:
        concepts = concepts[:2]
      
    concept_lists = extract_lists(concepts, listnums)


    for concept in tqdm(concepts, desc="Concept-List ", 
                        total=len(concepts)*2):
        for listnum in listnums:
            objects, labels = concept_lists[concept + "-" + listnum]

            if mock:
                objects = objects[:10]
                labels = labels[:10]

            preprime = construct_pre_prime(practice_perm=practice_perm)
            run_info = []

            # == TARGET BLOCK ==
            target_prompt = "I have Rule 1 in mind. I have labeled objects that follow Rule 1 True,"+\
                                "and objects that do not follow Rule 1 False.\n" +\
                                "Here are examples of labels based on Rule 1.\n"
            target_bookend = "\nWhat's the label of the last object based on Rule 1?"

            for n in tqdm(range(2, len(objects)), desc=f"{listnum} Num Examples "):
                target_block, target_info = make_experiment_block_from_list(
                    n_examples = n, 
                    rule = concept,
                    labels =labels,
                    objects=objects,
                    list_label=listnum,
                    prompt=target_prompt, 
                    bookend=target_bookend)
                
                messages = preprime + target_block
                reply, logprobs, _ = get_chatcompletion(messages, mock=mock)
                    
                target_info['query_repeated'] = objects[n-1] in set(objects[:n-1])
                target_info['model_answer'] = extract_answer(reply)
                target_info['full_output'] = reply
                target_info['full_logprobs'] = str(logprobs)

                run_info.append(target_info)

            pd.DataFrame.from_dict(run_info).to_csv(os.path.join(output_dir, f"{uid}_{listnum}_{concept}.csv"))
            with open(os.path.join(output_dir, f"logs/{uid}_{listnum}_{concept}_log.txt"), 'w+') as f:
              f.write(str(messages))

    print("Done!")


if __name__ == "__main__":

    timecode = str(time.time()).split(".")[0][-5:]

    MOCK = False
    mock_tag = "MOCK_" if MOCK else ""

    df = run_lists(uid=mock_tag+"gpt4_explanations_fol",
                concepts=SUBSETS['fol'],
                 mock=MOCK,  
                 log=True,
                 practice_perm=int(timecode)%500)



