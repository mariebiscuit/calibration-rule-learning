import json
import os, sys, re
import pickle
import pandas as pd
from typing import List, Dict, Tuple, Type
from collections import defaultdict

from custom_data import Obj

"""
Utilities to turn PTG16's SFL data—both human data and rule exemplar lists,
given as tsv files, into Pandas DataFrames or dictionaries

Functions are called in utilities that make data, 
i.e. utils.make_dataasets and utils.custom_data
"""

def format_shape(string: str) -> str:
    """
    :param str string: object raw representation e.g. "triangle,yellow,2"
    :returns:
        - object pretty string representation e.g. "medium yellow triangle"
    """
    shape, color, size = string.split(",")
    return " ".join([['small', 'medium', 'large'][int(size) - 1], color, shape])

def get_boolean_concepts() -> List[str]:
    """
    :returns:
        - list of rule codes for boolean rules
    """
    with open('data/boolean.txt', 'r') as f:
        boolean_concepts = [x.strip() for x in f.readlines()]

    return boolean_concepts

def make_rules_readable(rules: List[str]) -> List[str]:
    """
    Converts a list of rule codes to readable strings
    :param list(str) rules: list of rule codes e.g. ['hg04', 'hg05', ...]

    :returns: 
        - list of rules converted to readable names e.g. ['blue and medium', 'blue or medium'...]
    """
    with open('./data/labels_to_readable.json', 'r') as f:
        labels_to_readable = json.load(f)

    return [labels_to_readable[rule] for rule in rules]

def process_concept_folders(folder_path: str) -> Tuple[Dict, Dict]:
    """
    Takes a folder of concept txts from PTG16 and converts them to dictionaries. Folder of concept txts:
    (https://github.com/piantado/Fleet/tree/master/Models/GrammarInference-SetFunctionLearning/preprocessing/concepts)

    :param str folder_path: Path to concept folder
    
    :returns: 
        - concept_to_lambda: dictionary mapping from concept to lambda string
        - concept_to_data: dictionary mapping from concept to list num, to a dictionary containing sets and answers
            {'hg01': {'L1': {
                                'sets': [["circle, yellow, 3", "triangle,green,1"], ["circle,blue,1"]], 
                                'answers': [["True", "True"], ["True"]]
                            },
                    'L2': {...}
                    },
            "hg04" {...} ...}
    """
    concept_to_data = defaultdict(dict)
    concept_to_lambda = {}

    for filename in os.listdir(folder_path):
        _, concept, _, _, listnum =  filename.split("_")

        concept_to_data[concept][listnum] = {'sets': [], 'answers': []}
        with open(os.path.join(folder_path, filename)) as f:
            lines = f.readlines()

            concept_to_lambda[concept] = lines[0]

            setNumber = 1
            for l in lines[1:]:
                l = l.strip()
                parts = re.split(r"\t", l)
                
                set_objects = [",".join(x.split(",")[:3]) for x in parts[1:]]
                answers = [v == "#t" for v in re.findall(r"\#f|\#t", parts[0])]

                concept_to_data[concept][listnum]['sets'].append(set_objects)
                concept_to_data[concept][listnum]['answers'].append(answers)

    return concept_to_lambda, concept_to_data


def process_response_file(file_path: str) -> Tuple[Type[pd.DataFrame], Type[pd.DataFrame]]:
    """
    Processes individual human subject data into dataframe

    :param str file_path: filepath to .txt file containing individual responses

        subject	concept.number	total.concepts.done	concept	list	set.number	response.number	display.order	response	right.answer	time.since.start	instruction.reading.time	subject.age	subject.language	subject.education	subject.gender	subject.country	subject.MTurkHours	subject.puzzles	subject.MultipleHITs	click.yes	click.no	completed.concept
        aaa	1	0	hg30	L2	1	4	NA	F	F	0	67	21	"english"	CollegeDegree	Female	US	0to4	Yes	No	"NA"	"16777"	F	
    
    :returns: 
        - {"L1": list1_df, "L2": list2_df} dictionary of panda DFs where each DF has the structure
            concept | subject | object           | subject_order | set | num_in_set | answer | response|
            --------|---------|-------------------|-------------|------|------------|-------|---------|
            hg30    | aaa      | circle, blue, 1 |    3          |  1  | 1          | T      | F     |
            
            (objects in canonical list order, not subject response order)
    """ 

    generate_dict = lambda : {k: [] for k in ["concept","subject","object","subject_order", "set","num_in_set","answer","response"]}
    df_dicts = {"L1": generate_dict(), "L2": generate_dict()}

    with open('./data/labels_to_data.json', 'r') as f:
        stimuli_data = json.load(f)

    current_concept_subject = None
    current_idx = 0
    obj_idx_list = []

    with open(file_path, 'r') as file:
        file.readline()
        for line in file:
            line = line.strip().split()
            subject = line[0]    
            concept = line[3]
            listnum = line[4]
            setnum = int(line[5])
            responsenum = int(line[6])
            response = line[8]
            answer = line[9]

            if concept+subject+str(setnum) != current_concept_subject:
                obj_idx_list = []
                current_idx = 0
                current_concept_subject = concept+subject+str(setnum)

            current_idx +=1
            
            obj = stimuli_data[concept][listnum]['sets'][setnum - 1][responsenum - 1]
            obj = format_shape(obj)

            df_dict = df_dicts[listnum]
            df_dict['concept'].append(concept)
            df_dict['subject'].append(subject)
            df_dict['object'].append(obj)
            df_dict['subject_order'].append(current_idx)
            df_dict['set'].append(setnum)
            df_dict['num_in_set'].append(responsenum)
            df_dict['answer'].append(answer)
            df_dict['response'].append(response)

    output_dfs = {} 
    for listname, df_dict in df_dicts.items():
        output_df = pd.DataFrame.from_dict(df_dict)
        output_df = output_df.sort_values(['concept', 'subject', 'set', 'num_in_set']).reset_index(drop=True)
        output_df['item_num'] = output_df.groupby(['concept', 'subject']).cumcount() + 1
        output_dfs[listname] = output_df

    return output_dfs

def process_human_data_file_to_df(file_path: str, process_listnum="L2"):
    """
    Processes aggregate human subject data into dictionary of concept -> list of objects
    :param str file_path: str containing a .txt file of the format
        ```
        hg13 L2 1 2 False rectangle blue 3 11 13
        ```
    :returns:
        - pd.DataFrame of human results formatted as .csv
            ```
            ,concepts,answers,hyes,hno,example_num,obj
            0,hg13,False,8,16,1,medium blue triangle
            1,hg13,False,11,13,2,large blue rectangle
            ```

    """
    human_scores = defaultdict(dict)

    with open(file_path, 'r') as f:
        current_concept = None
        example_nums = []
        objs = []
        hyeses = []
        hnoes = []
        answers = []
        curent_objnum = 1
        for i, line in enumerate(f):
            concept, listnum, setnum, responsenum, answer, shape, color, size, hyes, hno = line.split(' ')
            
            if listnum == process_listnum:
                if current_concept is None:
                    current_concept = concept
                    current_objnum = 1

                if concept != current_concept:
                    human_scores[current_concept] = {'example_num': example_nums,
                                                    'obj': objs,
                                                    'hyes': hyeses,
                                                    'hno': hnoes,
                                                    'answers': answers}
                    current_concept = concept
                    current_objnum = 0
                    example_nums = []
                    objs = []
                    hyeses = []
                    hnoes = []
                    answers = []

                example_nums.append(current_objnum)
                objs.append(f"{['small', 'medium', 'large'][int(size) - 1]} {color} {shape}")
                hyeses.append(hyes)
                hnoes.append(hno.strip())
                answers.append(answer)
                current_objnum += 1
        
        human_scores[current_concept] = {'example_num': example_nums,
                                                    'obj': objs,
                                                    'hyes': hyeses,
                                                    'hno': hnoes,
                                                    'answers': answers}


    df_dict = {'concepts': [], 'answers': [], 'hyes': [], 'hno': [], 'example_num': [], 'obj': []}
    for concept in human_scores.keys():
        df_dict['answers'] += human_scores[concept]['answers']
        df_dict['hyes'] += human_scores[concept]['hyes']
        df_dict['hno'] += human_scores[concept]['hno']
        df_dict['obj'] += human_scores[concept]['obj']
        df_dict['example_num'] += human_scores[concept]['example_num']
        df_dict['concepts'] += [concept] * len(human_scores[concept]['obj'])
    
    df = pd.DataFrame.from_dict(df_dict)
    return df

def process_human_data_file_to_dict(file_path: str):
    """
    Processes aggregate human subject data into dictionary of concept -> list of objects
    Input:
    - file_path: str containing a .txt file of the format
    
        hg13 L2 1 2 False rectangle blue 3 11 13

    Output:
    - (train_data: Dict, val_data: Dict) of the format
        {'hg01': [('rectangle', 'blue', 'small, 0, False, 8, 11), 
                ('circle', 'yellow', 'medium, 0, False, 3, 16)
                ...],
        'hg03': [...],
        ...
        }
    """
    train_data = {}
    val_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split()
            concept = line[0]
            setnum = line[2]
            listnum = line[1]
            answer = line[4]
            shape = line[5]
            color = line[6]
            size = ['small', 'medium', 'large'][int(line[7])- 1]
            hyes = line[8]
            hno = line[9]

            if listnum == "L1":
                if concept not in train_data:
                    train_data[concept] = []
                train_data[concept].append((shape, color, size, setnum, answer, hyes, hno))
            else:
                if concept not in val_data:
                    val_data[concept] = []
                val_data[concept].append((shape, color, size, setnum, answer, hyes, hno))

    return train_data, val_data



if __name__ == "__main__":
    # process_human_data_file_to_df('./data/data.txt', "L2").to_csv('./data/compiled_all_humans.csv')
    _, l2_df = process_response_file('./data/TurkData-Accuracy.txt')
    l2_df.to_csv('./data/compiled_individual_humans.csv')

#     print(process_concept_folders('../bayesian_metamodel_exp/fleet/preprocessing/concepts'))