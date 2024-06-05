from typing import Tuple, List, Dict, Type
import json
import os
from tqdm import tqdm
import pandas as pd
import copy
import re

from datasets import Dataset

from utils.preprocess import format_shape
from utils.api_utils import get_together_completion, get_chatcompletion
from utils.get_concept_subsets import SUBSETS, READABLE

"""
## Experiment 1 and 2
Functions to perform inference on API models
"""

def query_together_completion(model_id: str, 
                                dataset: Type[Dataset], 
                                uid: str,
                                pre_answer_token: str = "->", 
                                get_logprobs: bool=False):

    if not os.path.isdir(f'./results/raw_results/{uid}'):
        os.mkdir(f'./results/raw_results/{uid}')

    for concept in dataset['concepts']:
        output = {k: [] for k in ['top_ans', 'mass_ans', 'raw_true_mass', 'raw_false_mass']}
        concept_dataset = dataset.filter(lambda x: x['concepts'] == concept)
        query_pieces = dataset['text'].split("->")
        running_message = ""
        for piece in query_pieces:
            running_message += piece + "->"

            top, _, _ = get_together_completion(model_url, prompt, max_tokens=1, logprobs=0, n=1, echo=False)

            if get_logprobs:
                _, _, response = get_together_completion(model_url, prompt + " True", max_tokens=1, logprobs=1, n=1, echo=True)
                t_logprob = response['prompt'][0]['logprobs']['token_logprobs'][-1]
                _, _, response = get_together_completion(model_url, prompt + " False", max_tokens=1, logprobs=1, n=1, echo=True)
                f_logprob = response['prompt'][0]['logprobs']['token_logprobs'][-1]
            else:
                t_logprob = -1
                f_logprob = -1

            output['top_ans'].append(top)
            output['mass_ans'].append(t_logprob > f_logprob)
            output['raw_true_mass'].append(t_logprob)
            output['raw_false_mass'].append(f_logprob)
        
        output_df = pd.DataFrame.from_dict(output)
        output_df.to_csv(f'./results/raw_results/{uid}/{uid}_{concept}.csv')
        
    print("Done evaluating!")
    
def process_reply(objects, reply):
    answers = []
    for obj in objects:
        objstring = format_shape(obj)
        regex = f"{objstring[1:]}" + "\s*(?:->|:)\s*(\w+)" + r"\b"
        match = re.search(regex, reply)
        if match is not None:
            answers.append(match.group(1))
        else:
            answers.append("None")
    return answers


def get_chat_inputs(data: Tuple[List[List], List[List]]):
    instruction = "Learn the secret rule to label the objects in groups correctly. " +\
        "The rule may consider the color, size and shape of objects, and may also consider the other objects in each group. " +\
        "If an object in a group follows the rule, it should be labeled 'True'. " +\
        "Otherwise it should be labeled 'False'. Be concise and always follow the arrow '->' with a object's label.\n\n"

    all_inputs = []
    running_message = [{"role": 'system', "content":  instruction}] 
    i = 0

    for objects, answers in zip(*data):
        running_message.append(
                {'role': 'user',
                'content': f"Label the following objects in Group {i+1}:\n" +\
                          "\n".join(["-" + format_shape(obj) for obj in objects])
                
                 })

        all_inputs.append(copy.deepcopy(running_message))

        running_message.append(
                {'role': 'assistant',
                'content': f"The labels for the objects in Group {i+1} are:\n" +\
                           "\n".join(["-" + format_shape(obj) + f"-> {answer}" for obj, answer in zip(objects, answers)])
                })
        i += 1

    return all_inputs

def get_chat_explanations(data: Tuple[List[List], List[List]]):
    instruction = "Learn the secret rule to label the objects in groups correctly. " +\
        "The rule may consider the color, size and shape of objects, and may also consider the other objects in each group. " +\
        "If an object in a group follows the rule, it should be labeled 'True'. " +\
        "Otherwise it should be labeled 'False'." +\
        "Begin your answer with 'Based on your examples, I've learned that...' and " +\
        "give your best concise description of the rule without asking for more evidence. " +\
        "Make sure your rule is consistent with the labels for all objects so far. " +\
        "Be concise and always follow the arrow '->' with an object's label. Label ALL objects in the given group.\n\n"

    all_inputs = []
    running_message = [{"role": 'system', "content":  instruction}] 
    i = 0

    for objects, answers in zip(*data):

        query_message = copy.deepcopy(running_message)
        query_message[-1]['content'] +=  f"\n\n Now concisely describe the labeling rule based on these groups of objects,"+\
                        f" then label all the following objects in Group {i+1}:\n" +\
                          "\n".join(["-" + format_shape(obj) for obj in objects])

        all_inputs.append(copy.deepcopy(query_message))

        running_message[-1]['content'] += f"\nThe labels for the objects in Group {i+1} are:\n" +\
                           "\n".join(["-" + format_shape(obj) + f"-> {answer}" for obj, answer in zip(objects, answers)])

        i += 1

    return all_inputs


def query_chatcompletion(model_id: str, 
                        output_folder: str,
                        raw_data: Dict, uid: str, log:str = None,
                        setstop: int = 25,
                        inputs_fn=get_chat_inputs):
    """
    :param Dict raw_data: Needs input generated by `process_concept_folders` from `preprocess.py`
    """
    output_fp = os.path.join(output_folder, f'{uid}')
    if not os.path.isdir(output_fp):
        os.mkdir(output_fp)

    for concept in tqdm(list(raw_data.keys()), total=len(list(raw_data.keys()))):
        replies = []
        top_ans = []
        data = (raw_data[concept]['sets'][:setstop], raw_data[concept]['answers'][:setstop])

        chat_inputs = inputs_fn(data)
        for i, chat_input in enumerate(chat_inputs):
            num_objects_in_set = len(chat_input[-1]['content'].split("Now concisely describe")[-1].split("-")) - 1 
            reply, _, _ = get_chatcompletion(chat_input, model=model_id)
            replies += [reply] * num_objects_in_set # duplicate reply for all objects in set
            top_ans += process_reply(data[0][i], reply) # get list of replies for each object

        print("Chat_inputs", len(chat_inputs))
        print("Replies", len(replies))
        print("Top answers", len(top_ans))
        
        column_dict = {
            'concept_num': [concept] * len(replies),
            'concept': [READABLE[concept]] * len(replies),
            'object': [format_shape(x) for l in data[0] for x in l],
            'answer': [x for l in data[1] for x in l],
            'model_answer': top_ans,
            'model_reply': replies,
            }

        output_df = pd.DataFrame.from_dict(column_dict)
        output_df.to_csv(os.path.join(output_fp, f'{uid}_{concept}.csv'))

        if log is not None:
            with open(log, 'a+') as f:
                for chat_input in chat_inputs:
                    json.dump(chat_input, f)
                    f.write("\n")

    print("Done evaluating!")

if __name__ == "__main__":

    import time
    import datasets

    timestamp =  str(time.time()).split(".")[0][-5:]

    """
    Sample of querying GPT4 for Experiment 1
    """
    # with open("./data/labels_to_data.json", 'r') as f:
    #     all_data = json.load(f)

    # L2_data = {k: d["L2"] for k, d in all_data.items() if k in ["hg02"]}
    # query_chatcompletion("gpt-4-1106-preview", L2_data, uid="gpt4_test", setstop=2)

    """
    Sample of querying GPT4 for Experiment 2
    """
    with open("./data/labels_to_data.json", 'r') as f:
        all_data = json.load(f)

    L2_data = {k: d["L2"] for k, d in all_data.items() if k in SUBSETS['boolean']}
    UID = "gpt4_boolean"
    query_chatcompletion("gpt-4-1106-preview", "./results/experiment_2/raw_results", L2_data, 
                        uid=UID, setstop=25, 
                        log=f"./results/experiment_2/raw_results/{UID}/{UID}_log.txt", inputs_fn=get_chat_explanations)