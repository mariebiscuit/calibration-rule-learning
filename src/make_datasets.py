import json
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from typing import Type, Dict, Tuple
import copy
import torch
import sys
import pickle

from preprocess import process_human_data_file_to_dict, process_response_file, format_shape

AGG_DATA_PROMPT =\
"""# Description
In this experiment, human subjects were instructed to learn the secret rule to label objects in groups correctly.
Subjects were told that rules may consider the color, size and shape of objects, and may also consider the other objects in each group.
If an object in a group follows the rule, subjects were to label it 'True'. Otherwise subjects were to label it 'False'.
Subjects were shown the correct labels for the objects after every group, and asked to learn from their errors.
The results below report the proportion of subjects that labeled a given object in the group 'True'.
For reference, the correct labels for each object are also shown.

# Results"""

LABEL_DATA_PROMPT = "# Instructions\n" +\
        "Learn the secret rule to label the objects in groups correctly. " +\
        "The rule may consider the color, size and shape of objects, and may also consider the other objects in each group. " +\
        "If an object in a group follows the rule, it should be labeled 'True'. " +\
        "Otherwise it should be labeled 'False'.\n\n" +\
        "# Quiz\n\n"

ROLEPLAY_DATA_PROMPT = """In this particular study, {name} was told to learn the secret rule to label the objects in groups correctly. 
The rule could consider the color, size and shape of objects, and could also consider the other objects in each group.
If an object in a group follows the rule, {name} should label it 'True'. Otherwise it should be labeled 'False'.
After labeling a group of objects, {name} was given feedback on what the correct labels were. Here's how it went."""

def __make_agg_set_message(setnum, current_objs, current_yesrate, current_ans=None):
    if current_ans is None:
        p = f"""## Group {setnum} Objects\n"""
        p += "\n".join(current_objs) + "\n\n"
        p += f"""## Group {setnum} Proportion Subjects Answered True\n"""
        p += "\n".join([f"{o} -> {a}" for o, a in zip(current_objs, current_yesrate)]) + "\n\n"
        return p
    else:
        p = f"""## Group {setnum} Objects\n"""
        p += "\n".join(current_objs) + "\n\n"
        p += f"""## Group {setnum} Proportion Subjects Answered True\n"""
        p += "\n".join([f"{o} -> {a}" for o, a in zip(current_objs, current_yesrate)]) + "\n\n"
        p += f"""## Group {setnum} Correct Labels\n"""
        p += "\n".join([f"{o}: {a}" for o, a in zip(current_objs, current_ans)]) + "\n\n"
        return p

def __make_agg_set_queries(setnum, current_objs, current_ans):
    paragraphs = []
    p = f"""## Group {setnum}\n"""
    p += "\n".join(current_objs) + "\n\n"
    p += f"""## Group {setnum} Proportion Answered True\n"""
    for o, a in zip(current_objs, current_ans):
        p += f"{o} -> "
        paragraphs.append(p)
        p += f"{a}\n"
    p+= "\n" 
    return paragraphs

def make_agg_data(file_path: str, 
                prompt:str = AGG_DATA_PROMPT, 
                with_labels: bool = True) -> Dict:
    """
    :param str file_path: data that looks like
        hg13 L2 1 2 False rectangle blue 3 11 13
    :param str prompt: prompt text to prepend
    :param bool with_labels: whether labels for data should be appended to the training data

    :returns:
        - Dictionary with ['train', 'val'] keys where each dictionary contains a Dataset object with following fields
        
            concepts | text | objs                                           | yesrates          | answers |
            ---------| ------| ----------------------------------------------| -------------------| -----------------|
            hg03     | "..." | [large blue triangle, small yellow circle...]]|  [0.625, 0.208...] | [False, True...] |

            'text' is the following prompt for all objects in a concept: 
                ```
                ## Group 1 Objects
                large blue triangle
                small yellow circle

                ## Group 1 Proportion Subjects Answered True
                large blue triangle -> 0.625
                small yellow circle -> 0.208
                    
                [if with_labels is true]
                ## Group 1 Correct Labels
                large blue triangle: False
                small yellow circle: True
                ```
    """
                    
    train_data, val_data = process_human_data_file_to_dict(file_path)
    splits = {'L1': train_data, 'L2': val_data}
    outputs = {}
    for j, (split, data) in enumerate(splits.items()):
        paragraphs = [] # list of lists of (para, response)
        concepts = []
        obj_lists = []
        yesrate_lists = []
        answer_lists = []
        for i, (concept, objects) in enumerate(data.items()):
            paragraph = prompt
            objs = []
            answers = []
            yesrates = []
            
            current_setnum = None
            current_objects = []
            current_yesrates = []
            current_answers = []
            for j, obj in enumerate(objects):
                shape, color, size, setnum, answer, hyes, hno = obj
                total_participants = int(hyes) + int(hno)
                yesrate = "{:.3f}".format(int(hyes) / total_participants)

                if current_setnum is None:
                    current_setnum = setnum 

                if (current_setnum != setnum) or (j == (len(objects)-1)):
                    #flush
                    if with_labels:
                        paragraph += __make_agg_set_message(current_setnum, current_objects, current_yesrates, current_answers)
                    else:
                        paragraph += __make_agg_set_message(current_setnum, current_objects, current_yesrates)

                    objs.append(copy.deepcopy(current_objects))
                    answers.append(copy.deepcopy(current_answers))
                    yesrates.append(copy.deepcopy(current_yesrates))

                    current_answers = []
                    current_objects = []
                    current_yesrates = []
                    current_setnum = setnum

                current_objects.append(f"{size} {color} {shape}")
                current_yesrates.append(yesrate)
                current_answers.append(answer)
            
            concepts.append(concept)
            paragraphs.append(paragraph)
            obj_lists.append(objs)
            answer_lists.append(answers)
            yesrate_lists.append(yesrates)
    
        outputs[split] = {'concepts': concepts, 
                        'text': paragraphs, 
                        'objs': obj_lists, 
                        'yesrates': yesrate_lists, 
                        'answers': answer_lists}

    # return {split: Dataset.from_dict(outputs[split]) for split in ['train', 'val']}
    return DatasetDict({"L1": Dataset.from_dict(outputs["L1"]), "L2": Dataset.from_dict(outputs["L2"])})

def make_label_data_kl(file_path: str, prompt:str=LABEL_DATA_PROMPT) ->\
     Tuple[Dict[str, Type[Dataset]], Dict[str, Type[torch.Tensor]]]:
    """
    Generates dataset where object set is shown and then actual answers are filled in.
    And returns a matching tensor of the human distribution over T/F.

    :param str file_path: filepath to data that looks like
        ```
        hg13 L2 1 2 False rectangle blue 3 11 13
        ```
    :param str prompt: prompt string to prepend

    :returns:
        - outputs: Dictionary with ['train', 'val'] keys where each dictionary contains a Dataset object with fields
        idx | concepts | text  | objs                                            | answers         |
        ---| ---------| ------| -------------------------------------------------| -----------------|
        0   | hg03     | "..." | [ large blue triangle, small yellow circle ...] |  [False, True...] |

            And 'text' is the following prompt for all objects in the concept (with actual labels):
            ```
            ## Group 1
            large blue triangle
            small yellow circle

            ## Group 1 Answers
            large blue triangle -> False
            small yellow circle -> True
            ```
    """

    train_data, val_data = process_human_data_file_to_dict(file_path)
    splits = {'L1': train_data, 'L2': val_data}
    yesrate_outputs = {}

    outputs = {}
    for j, (split, data) in enumerate(splits.items()):
        paragraphs = [] # list of lists of (para, response)
        concepts = []
        obj_lists = []
        yesrate_lists = []
        answer_lists = []
        for i, (concept, objects) in enumerate(data.items()):
            paragraph = prompt
            objs = []
            answers = []
            yesrates = []
            
            current_setnum = None
            current_objects = []
            current_yesrates = []
            current_answers = []
            for j, obj in enumerate(objects):
                shape, color, size, setnum, answer, hyes, hno = obj
                total_participants = int(hyes) + int(hno)
                yesrate = int(hyes) / total_participants

                if current_setnum is None:
                    current_setnum = setnum 

                if (current_setnum != setnum):
                    #if changed set, flush
                    paragraph += f"""## Group {current_setnum}\n"""
                    paragraph += "\n".join(current_objects) + "\n\n"
                    paragraph += f"""## Group {current_setnum} Answers\n"""
                    paragraph += "\n".join([f"{o} -> {a}" for o, a in zip(current_objects, current_answers)]) + "\n\n"

                    objs.append(copy.deepcopy(current_objects))
                    answers.append(copy.deepcopy(current_answers))
                    yesrates += copy.deepcopy(current_yesrates)

                    current_answers = []
                    current_objects = []
                    current_yesrates = []
                    current_setnum = setnum

                current_objects.append(f"{size} {color} {shape}")
                current_yesrates.append([yesrate, 1-yesrate])
                current_answers.append(answer)

                if (j == (len(objects)-1)):
                    # if last object, flush
                    paragraph += f"""## Group {current_setnum}\n"""
                    paragraph += "\n".join(current_objects) + "\n\n"
                    paragraph += f"""## Group {current_setnum} Answers\n"""
                    paragraph += "\n".join([f"{o} -> {a}" for o, a in zip(current_objects, current_answers)]) + "\n\n"

                    objs.append(copy.deepcopy(current_objects))
                    answers.append(copy.deepcopy(current_answers))
                    yesrates += copy.deepcopy(current_yesrates)
            
            concepts.append(concept)
            paragraphs.append(paragraph)
            obj_lists.append(objs)
            answer_lists.append(answers)
            yesrate_lists.append(torch.Tensor(yesrates))

        yesrate_outputs[split] = yesrate_lists
        outputs[split] = {'idx': np.arange(len(concepts)), 
                        'concepts': concepts, 
                        'text': paragraphs, 
                        'objs': obj_lists, 
                        'answers': answer_lists,
                        'hdist': yesrate_lists}

    return DatasetDict({"L1": Dataset.from_dict(outputs["L1"]), "L2": Dataset.from_dict(outputs["L2"])}), yesrate_outputs

    # return DatasetDict({"L1"})
    # {split: Dataset.from_dict(outputs[split]) for split in ['train', 'val']}

        
def make_indiv_rp_data(filepath: str, 
                        prompt:str=ROLEPLAY_DATA_PROMPT, 
                        cached_data_fp=None):
    """
    Generates dataset where model is asked to 'roleplay' a specific participant that has some fictional bio generated

    :param str filepath: filepath to txt containing individual subject responses that looks like
        subject	concept.number	total.concepts.done	concept	list	set.number	response.number	display.order	response	right.answer	time.since.start	instruction.reading.time	subject.age	subject.language	subject.education	subject.gender	subject.country	subject.MTurkHours	subject.puzzles	subject.MultipleHITs	click.yes	click.no	completed.concept
        aaa	1	0	hg30	L2	1	4	NA	F	F	0	67	21	"english"	CollegeDegree	Female	US	0to4	Yes	No	"NA"	"16777"	F	
    :param prompt str: prompt text to prepend
    """
    data, _ = process_response_file(filepath)
    data = data.sort_values(['subject', 'concept', 'subject_order']).dropna()
    subject_df = pd.read_csv('./data/generated_subject_bios.csv', index_col=0)

    paragraphs = []
    concepts = []
    subjects = []

    for group in data.groupby(['subject', 'concept']):
        paragraph = ""
        uid, concept = group[0]
        bio = subject_df[subject_df['subject'] == uid]['bio'].iloc[0]
        name = bio.split(" ")[0]
        paragraph += bio + "\n" + prompt.format(name=name)

        for sets in group[1].groupby(['set']):
            setnum = sets[0][0]
            objects = sets[1]['object'].apply(lambda x: format_shape(x)).tolist()
            responses = sets[1]['response'].apply(lambda x: {"T": "True", "F": "False"}[x]).tolist()
            answers = sets[1]['answer'].apply(lambda x: {"T": "True", "F": "False"}[x])
            paragraph += f"\n\n## {name} was shown Group {setnum}...\n" + "\n".join(objects)
            paragraph += f"\n\n## {name} labelled Group {setnum} as...\n"
            paragraph += "\n".join([f"{o}-> {a}" for o, a in zip(objects, responses)])
            paragraph += f"\n\n## {name} was then told that the right answers for Group {setnum} were...\n"
            paragraph += "\n".join([f"{o}: {a}" for o, a in zip(objects, answers)])
        
        paragraphs.append(paragraph)
        subjects.append(uid)
        concepts.append(concept)
    
    outdict = {'concepts': concepts, 'subjects': subjects, 'text': paragraphs}
    return Dataset.from_dict(outdict)

if __name__ == "__main__":

   datasets, yesrates = make_label_data_kl("./data/data.txt")
   datasets.save_to_disk('./datasets/kl_dataset')
   datasets['L1'].select([1,2,3]).to_csv('./datasets/kl_dataset/sample_L1_kl_dataset.csv')
   datasets['L2'].select([1,2,3]).to_csv('./datasets/kl_dataset/sample_L2_kl_dataset.csv')

   with open('./datasets/kl_dataset/L1_hdists.pkl', 'wb') as f:
        pickle.dump(yesrates['L1'], f)
    
   with open('./datasets/kl_dataset/L2_hdists.pkl', 'wb') as f:
        pickle.dump(yesrates['L2'], f)
