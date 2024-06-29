import pandas as pd
from typing import Type
import sys, os
import re
from collections import defaultdict
from tqdm import tqdm
import json
import traceback
import numpy as np
import copy

from utils.api_utils import get_chatcompletion
from utils.preprocess import format_shape

"""
=== Experiment 2 ===
Code to process GPT-4's natural language explanation into Python code for 
evaluation of previous objects
"""

shapes = ['star', 'oval', 'rectangle', 'circle', 'triangle', 'square']
colors = ['green', 'blue', 'yellow', 'purple', 'orange', 'red']
sizes = ['small', 'medium', 'large']
patterns = ['dotted', 'striped', 'solid']
feature_map = {shape: 'shape' for shape in shapes}
feature_map.update({color: 'color' for color in colors})
feature_map.update({size: 'size' for size in sizes})
feature_map.update({pattern: 'pattern' for pattern in patterns})

class dotdict(dict):
    # https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """dot.notation access to dictionary attributes"""
    def __init__(self, d):
        self.attrs = d

    def __getattr__(self, x):
        return self.attrs[x]
        # return y if y is not None else False

    def __repr__(self):
        return "dotdict(" + self.attrs.__repr__() + ")"

    def __eq__(self, other):
        isClass = isinstance(other, self.__class__)
        if not isClass:
            return False

        for key, value in self.attrs.items():
            if other.attrs[key] != value:
                return False
        
        return True

class objdotdict(dotdict):
    # https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    def __init__(self, d):
        super(objdotdict, self).__init__(d)

        for k, _ in list(d.items()):
            if k in feature_map:
                self.attrs[feature_map[k]] = k

        self.attrs['size_num'] = {'small': 1, 'medium': 2, 'large': 3}[self.attrs['size']]
        self.attrs['medium_sized'] = d['medium']
        self.attrs['small_sized'] = d['small']
        self.attrs['large_sized'] = d['large']
        self.attrs['any_size'] = True
        self.attrs['any_shape'] = True
        self.attrs['any_color'] = True
        self.attrs['has_color'] = True
        self.attrs['circular'] = d['circle']
        self.attrs['triangular'] = d['triangle']
        self.attrs['rectangular'] = d['rectangle']
    
def process_python_code(code: str):
    try:
        statement = code.split("return ")[1]
        statement = re.sub(r'(?:\w+|\w+\.\w+) == (?:\"|\')(\w+)(?:\"|\')', r"\1", statement)
        statement = re.sub(r'circular', 'circle', statement)
        statement = re.sub(r'triangular', 'triangle', statement)
        statement = re.sub(r'rectangular', 'triangle', statement)
        statement = re.sub(r'\!\=', 'not', statement)
        statement = re.sub(r'(obj\.|drawing\.|drawings\.|object\.|shape\.|size\.|-sized|sized|shape|colored|has_color|color|size|\"|\')', '', statement)

        # Turning triangle.blue into triangle and blue.
        if re.sub(r'triangle\.', '', statement) != statement:
            statement = 'triangle and ' + re.sub(r'triangle\.', '', statement)
        elif re.sub(r'rectangle\.', '', statement) != statement:
            statement = 'rectangle and ' + re.sub(r'rectangle\.', '', statement)
        elif re.sub(r'circle\.', '', statement) != statement:
            statement = 'circle and ' + re.sub(r'circle\.', '', statement)
        
        # Removing dangling and/ or: e.g. yellow and (small and )
        statement = re.sub(r'and(\s[^\w\(])', r"\1", statement)
        statement = re.sub(r'\s+', ' ', statement)
        
    except:
        statement = "ERROR"
        
    return statement

def make_object(object_str: str, setnum: int=None, concept=None):
    d = defaultdict(lambda: False)
    for k in object_str.split(" "):
        d[k] = True

    if setnum is not None and concept is not None:
        with open('./data/labels_to_data.json', 'r') as f:
            data = json.load(f)
        d['current_set'] = dotdict({'objs': [make_object(format_shape(x)) for x in data[concept]['L2']['sets'][setnum-1]],
                              'answers': data[concept]['L2']['answers'][setnum]})
        d['past_sets'] = dotdict({'obj_sets': [[make_object(format_shape(x)) for x in s] for s in data[concept]['L2']['sets'][:setnum]],
                             'answer_sets': data[concept]['L2']['answers'][:setnum-1]})
    
    dotd = objdotdict(d)
    return dotd


def evaluate_python_code(code, target_query, setnum=None, concept=None):
    loc = {}
    try:
        obj = make_object(target_query, setnum, concept)
        exec(code + "\nanswer = is_rule(obj)", {'obj': make_object(target_query, setnum, concept), 
                                                'sized': 0,
                                                'np': np}, loc)
        loc['answer'] = bool(loc['answer'])
    except Exception as e:
        print(code, "\n", target_query, setnum, concept)
        print(traceback.format_exc())
        loc['answer'] = "ERROR"
    
    return loc['answer']


def rules_equal(hyp_exp: str, true_exp: str):
    """
    hyp_rule : green or triangle or circle
    true_exp: green_OR_triangle
    """

    true_exp = set(re.sub(r'(\(|\))', '', str(true_exp).lower()).split(" "))
    hyp_exp = set(re.sub(r'(\(|\))', '', str(hyp_exp).lower()).split(" "))
    return true_exp == hyp_exp

def translate_rules(df: Type[pd.DataFrame], mock=True):
    prompt = 'Translate this to a Python predicate function:\n'
    in_context_examples = [
        {'role': 'system', 
        'content': 'You are a natural language to code translator that turns a description of a rule into the predicate function in Python. Return only the python code, with no docstrings. The Python code should only be the function header that takes in `obj` as its only argument, and a return statement. Do not use any if or else statements.'},
        {'role': 'user',
        'content': f"{prompt}'Based on your examples, I've learned that Rule 1 is that objects must both be large and have a color that matches the shape in a specific way: circles are yellow, rectangles and triangles are blue.'"
       },
        {'role': 'assistant',
        'content': 
"""def is_rule(obj):
    return obj.large and ((obj.circle and obj.yellow) or (obj.rectangle and obj.blue) or (obj.triangle and obj.blue))"""
        },
        {'role': 'user',
        'content': f"{prompt}'Based on your examples, I've learned that Rule 2 might be that no objects of any size or color are labelled true, as there are no examples provided of any object labelled true. Since no pattern of true labels is provided, I must assume either no object meets the criteria for Rule 2, or not enough examples have been given to determine what constitutes a true label under Rule 2. Therefore, with the information provided, the label for the last object is:'"
        },
        {'role': 'assistant',
        'content': 
"""def is_rule(obj):
    return False"""
        },
        {'role': 'user',
        'content': f"{prompt}'Based on your examples, I've learned that Rule 2 is currently indeterminate with the given examples. None of the objects have been labeled True, so there’s no clear pattern that dictates why an object would be labeled True under Rule 2. Without any positive examples, it's impossible to deduce what makes an object follow Rule 2.'"
        },
        {'role': 'assistant',
        'content': 
"""def is_rule(obj):
    return None"""
        },
        {'role': 'user',
        'content': f"{prompt}'Based on your examples, I've learned that Rule 2 involves the obj being a blue shape or a large circle to be labeled True.'"
        },
        {'role': 'assistant',
        'content': 
"""def is_rule(obj):
    return obj.blue or (obj.large and obj.circle)"""
        }
    ]
    
    if mock:
        df = df.iloc[[2, 6], :]

    outputs = [] # Python code generated by GPT3.5
    exps = [] # Rule expression cleaned from Python code
    evals = [] # evaluation of Python code 
    coverage = [] # percentage of past examples that proposed rule explains
    sets = []

    last_set_boundary = -1
    current_setnum = 1
    df = df.reset_index(drop=True)
    for i, row in tqdm(df.iterrows(), desc="Target rules: ", total=len(df)):
        if (i > 0) and (df['model_reply'].iloc[i] != df['model_reply'].iloc[i-1]):
            current_setnum +=1
            last_set_boundary = i - 1

        sets.append(current_setnum)

        description = row['model_reply'].split("\n")[0]
        message = in_context_examples + [{'role': 'user', 'content': f"{prompt}" + "'" + f"{description}" + "'"}]
        reply, _, _ = get_chatcompletion(message, model='gpt-3.5-turbo-0125', mock=False)
        outputs.append(reply)
        exps.append(process_python_code(reply))
        evals.append(evaluate_python_code(reply, row['object']))

        backapply_answers = df['object'].iloc[:last_set_boundary+1].apply(lambda x: evaluate_python_code(reply, x))
        coverage.append((backapply_answers == df['answer'].iloc[:last_set_boundary+1]).astype(int).mean())

    print("Assembling df...")
    df['set'] = sets
    df['code'] = outputs
    df['rule_expression'] = exps
    df['rule_evaluation'] = evals
    df['rule_coverage'] = coverage

    df.loc[:, 'consistency'] = df['rule_evaluation'].astype(str) == df['model_answer'].astype(str)
    df.loc[:, 'rule_match'] = df.apply(lambda x: rules_equal(x['rule_expression'], x['concept']), axis=1)
    return df

def extract_answers(df, output_col):
    model_answer = []
    for i, row in df.iterrows():
        regex = r': (True|False)'
        answer = re.search(regex, row[output_col])
        model_answer.append(answer.group(1) if answer else "NA")

    return model_answer

def test_python_code_eval():
    test1 = """def is_rule(obj):
    return (obj.medium and obj.blue) and obj.circle"""
    test2= """def is_rule(obj):
    return False"""
    test3= """def is_rule(obj):
    return obj.blue or (obj.large and obj.circle)"""
    test4 = """def is_rule(obj):
    return obj.medium-sized"""
    test5 = """def is_rule(shape, color):
    return shape == "triangle" or (shape == "rectangle" and color == "blue")"""
    test6 = """def is_rule(shape):
    return shape.striped and shape.square"""
    test7 = """def is_rule(obj):
    return (obj.yellow and obj.size) or (obj.small and obj.color)"""

    # same color as an object in the set
    test8 = """def is_rule(obj):
    return np.any([x.color == obj.color for x in obj.current_set.objs])"""
    
    # object was previously true
    test9 = """def is_rule(obj):
    return np.any([(ob == obj) and an for objects, answers in zip(obj.past_sets.obj_sets, obj.past_sets.answer_sets) for ob, an in zip(objects, answers)])"""

    test10 = """def is_rule(obj):  
    return np.any([x.color == obj.color for s in obj.past_sets.obj_sets for x in s]) or np.any([x.shape == obj.shape for s in obj.past_sets.obj_sets for x in s]) or np.any([x.size == obj.size for s in obj.past_sets.obj_sets for x in s])"""

    test11 = """def is_rule(obj):
    return obj.medium and (obj.color == [o for s in obj.past_sets.obj_sets for o in s][np.nonzero(np.array([o for s in obj.past_sets.answer_sets for o in s]))[-1][-1]].color) and (obj.shape !=[o for s in obj.past_sets.obj_sets for o in s][np.nonzero(np.array([o for s in obj.past_sets.answer_sets for o in s]))[-1][-1]].shape)"""

    test12 = """def is_rule(obj):
    return obj.medium and ((len(np.nonzero(np.array([o for s in obj.past_sets.answer_sets for o in s]))[-1]) > 0) and (obj.color == [o for s in obj.past_sets.obj_sets for o in s][np.nonzero(np.array([o for s in obj.past_sets.answer_sets for o in s]))[-1][-1]].color) and (obj.shape !=[o for s in obj.past_sets.obj_sets for o in s][np.nonzero(np.array([o for s in obj.past_sets.answer_sets for o in s]))[-1][-1]].shape))"""
    # assert process_python_code(test1) == "(medium and blue) and circle"
    # assert process_python_code(test2) == "False"
    # assert process_python_code(test3) == "blue or (large and circle)"
    # assert process_python_code(test4) == "medium"
    # assert process_python_code(test5) == "triangle or (rectangle and blue)"
    # assert process_python_code(test7) == "(yellow ) or (small )"

    # assert str(evaluate_python_code(test4, 'medium yellow rectangle')) == str(True)
    # assert str(evaluate_python_code(test6, 'dotted red star')) == str(False)
    # assert str(evaluate_python_code(test8, 'medium yellow rectangle', 1, 'hg19')) == str(True)
    # assert str(evaluate_python_code(test9, 'large yellow triangle', 3, 'hg07')) == str(True)
    # assert str(evaluate_python_code(test9, 'small green rectangle', 2, 'hg07')) == str(False)
    # assert str(evaluate_python_code(test9, 'small yellow circle', 2, 'hg07')) == str(True)
    print(evaluate_python_code(test12, 'medium green circle', 1, 'hg47'))
    # assert str(evaluate_python_code(test11, 'small yellow circle', 2, 'hg07')) == str(True)

    o = make_object("large blue rectangle")
    assert o.large == True and o.blue == True and o.green == False
    assert o.color == "blue" and o.size == "large" and o.shape != 'circle'

    o = make_object("medium yellow rectangle")
    assert o.medium_sized == True and o.rectangular == True and o.circular == False and o.medium == True


if __name__ == "__main__":
    test_python_code_eval()

    # UID = "gpt4_fol"
    # FOLDER = f"./results/experiment_2/raw_results/{UID}"
    # output_folder = FOLDER + "_translated"

    # if not os.path.isdir(output_folder):
    #     os.mkdir(output_folder)
        
    # MOCK = False

    # dfs = []
    # for file in os.listdir(FOLDER):
    #     if file.endswith(".csv"):
    #         print(f"Working on {file}...")
    #         df = translate_rules(pd.read_csv(os.path.join(FOLDER, file)), mock=MOCK)
    #         df.to_csv(os.path.join(output_folder,file))