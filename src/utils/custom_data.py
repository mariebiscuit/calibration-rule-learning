from typing import Dict, List, Callable, Type, Tuple
from collections import defaultdict
import numpy as np
import pickle

class FeatureSet:
    def __init__(self, feature_dict):

        self.__attr_to_props = feature_dict
        self.attrs = list(self.__attr_to_props.keys())
        self.__props_to_attr = {prop: attr for attr in self.attrs for prop in self.__attr_to_props[attr]}

    def prop_to_attr(self, prop: str) -> str:
        return self.__props_to_attr.get(prop)

    def get_props(self, attr: str) -> List[str]:
        return self.__attr_to_props.get(attr)

    def get_attrs(self) -> List[str]:
        return self.attrs

paper_shape_features = FeatureSet(   
    {'size': ['small', 'medium', 'large'],
    'color': ['blue', 'green', 'yellow'],
    'shape': ['circle', 'rectangle', 'triangle']
    })

priming_pilot = FeatureSet(   
    {'size': ['small', 'medium', 'large'],
    'color': ['blue', 'green', 'yellow'],
    'shape': ['oval', 'square', 'star'],
    })

new_shape_features = FeatureSet({
    'fill': ['solid', 'dotted', 'striped'],
    'color': ['red', 'purple', 'orange'],
    'shape': ['oval', 'square', 'star'],
    })
animal_features = FeatureSet({
    'pattern': ['ticked', 'spotted', 'striped'],
    'color': ['brown', 'black', 'silver'],
    'animal': ['rabbit', 'dog', 'cat'],
    }
)

class Obj:
    """ Represent bundles of features"""
    def __init__(self, **f):
        self.attributes = []
        for k, v in f.items():
            setattr(self, k, v)
            self.attributes.append(k)

    # @staticmethod
    # def from_string(self, string: str, feature_set: FeatureSet, sep=" "):
    #     properties = string.split(sep)
    #     return cls({prop_to_attr[prop]: prop for prop in properties})

    def __eq__(self, other):
        return (set(self.attributes) == set(other.attributes)) and\
                all([self.__getattribute__(k) == self.__getattribute__(k) for k in self.attributes])

    def __hash__(self):
        return hash(tuple([self.__getattribute__(k) for k in self.attributes]))
    
    def __str__(self):
        outstr = '<OBJECT: '
        for k, v in self.__dict__.items():
            outstr = outstr + str(k) + '=' + str(v) + ' '
        outstr = outstr + '>'
        return outstr

    def __repr__(self): # used for being printed in lists
        return str(self)
    
    def format_str(self, order: List[str]=None, sep=" "):
        order = self.attributes if order is None else order
        return sep.join([self.__getattribute__(key) for key in order])


def check_equality(attr, prop):
    return lambda x: str((x.__getattribute__(attr) == prop))

def check_or(attr_1, prop_1, attr_2, prop_2):
    return lambda x: str(((x.__getattribute__(attr_1) == prop_1) | (x.__getattribute__(attr_2) == prop_2)))

def check_and(attr_1, prop_1, attr_2, prop_2):
    return lambda x: str(((x.__getattribute__(attr_1) == prop_1) & (x.__getattribute__(attr_2) == prop_2)))

def make_rule(rule_name: str, feature_set: Type[FeatureSet]):
    args = rule_name.split("_")
    if args[0] == "is":
        assert len(args) == 2, "Invalid number of arguments for predicate."
        prop = args[1]
        assert feature_set.prop_to_attr(prop) is not None, "Property attribute not found."
        attr =  feature_set.prop_to_attr(prop)

        return check_equality(attr, prop), rule_name
    else:
        assert len(args) == 3, "Invalid number of arguments for double."

        prop1, prop2 = args[0], args[2]
        assert feature_set.prop_to_attr(prop1) is not None, "Property attribute not found."
        assert feature_set.prop_to_attr(prop2) is not None, "Property attribute not found."

        attr1, attr2 =  feature_set.prop_to_attr(prop1),  feature_set.prop_to_attr(prop2)

        if args[1] == "OR":
            return check_or(attr1, prop1, attr2, prop2), rule_name
        elif args[1] == "AND":
            return check_and(attr1, prop1, attr2, prop2), rule_name
        else:
            raise ValueError("Invalid operator")

# Control: Doesn't involve "OR"
# Control: Balanced True/False?

def make_cross_or_rules(feature_set: Type[FeatureSet]):
    attrs = feature_set.get_attrs()
    rules = defaultdict()

    combinations = [(0, 1), (1, 2), (0, 2)]
    for comb in combinations:
        attr1, attr2 = attrs[comb[0]], attrs[comb[1]]
        attr1_fts, attr2_fts = feature_set.get_props(attr1), feature_set.get_props(attr2)

        for attr1_ft in attr1_fts:
            for attr2_ft in attr2_fts:
                rules[attr1_ft + '_OR_' + attr2_ft] = check_or(attr1, attr1_ft, attr2, attr2_ft)

    return rules

def get_rule_groups(feature_set: Type[FeatureSet]) -> Tuple[Dict[str, List], Dict[str, str]]:
    rule_groups = defaultdict()

    attrs = feature_set.get_attrs()
    for attr in attrs:
        rule_groups[f"{attr}"] = []

        for attr_ft in feature_set.get_props(attr):
            rule_groups[f"{attr}"].append(f"{attr_ft}")

    for attr1 in attrs:
        for attr2 in attrs:

            attr1_fts, attr2_fts = feature_set.get_props(attr1), feature_set.get_props(attr2)

            for operator in ["OR", "AND"]:
                rule_groups[f"{attr1}_{operator}_{attr2}"] = []
                for attr1_ft in attr1_fts:
                    for attr2_ft in attr2_fts:
                        rule_groups[f"{attr1}_{operator}_{attr2}"].append(f"{attr1_ft}_{operator}_{attr2_ft}")
        
    return rule_groups, {vi: k for k, v in rule_groups.items() for vi in v}


def make_cross_and_rules(feature_set: Type[FeatureSet]):
    attrs = feature_set.get_attrs()
    rules = defaultdict()

    combinations = [(0, 1), (1, 2), (0, 2)]
    for comb in combinations:
        attr1, attr2 = attrs[comb[0]], attrs[comb[1]]
        attr1_fts, attr2_fts = feature_set.get_props(attr1), feature_set.get_props(attr2)

        for attr1_ft in attr1_fts:
            for attr2_ft in attr2_fts:
                rules[attr1_ft + '_AND_' + attr2_ft] = check_or(attr1, attr1_ft, attr2, attr2_ft)

    return rules

def make_all_rules(feature_set: Type[FeatureSet]):
    attrs = feature_set.get_attrs()
    three_ft = len(attrs) == 3
    rules = defaultdict()
    
    # One Feature
    for attr in attrs:
        for prop in feature_set.get_props(attr):
            rules['is_' + prop] = check_equality(attr, prop)

    # Within features OR
    for attr in attrs:
        props = feature_set.get_props(attr)
        for i, _ in enumerate(props):
            new_props = [i for i in props]
            new_props.pop(i)
            prop_1, prop_2 = new_props
            rules[prop_1 + '_OR_' + prop_2] = check_or(attr, prop_1, attr, prop_2)

    # Across features AND and OR
    if three_ft:
        combinations = [(0, 1), (1, 2), (0, 2)]
        for comb in combinations:
            attr1, attr2 = attrs[comb[0]], attrs[comb[1]]
            attr1_fts, attr2_fts = feature_set.get_props(attr1), feature_set.get_props(attr2)

            for attr1_ft in attr1_fts:
                for attr2_ft in attr2_fts:
                    rules[attr1_ft + '_AND_' + attr2_ft] = check_and(attr1, attr1_ft, attr2, attr2_ft)
                    rules[attr1_ft + '_OR_' + attr2_ft] = check_or(attr1, attr1_ft, attr2, attr2_ft)

    return rules


def make_all_objects(feature_set: Type[FeatureSet]):
    all_shapes = []

    attrs = feature_set.get_attrs()
    three_ft = len(attrs) == 3
    
    if three_ft:
        for attr1_ft in feature_set.get_props(attrs[0]):
            for attr2_ft in feature_set.get_props(attrs[1]):
                for attr3_ft in feature_set.get_props(attrs[2]):
                    kwargs = {attrs[0]: attr1_ft, attrs[1]: attr2_ft, attrs[2]: attr3_ft}
                    all_shapes.append(Obj(**kwargs))
    else:
        for attr1_ft in feature_set.get_props(attrs[0]):
            for attr2_ft in feature_set.get_props(attrs[1]):
                    kwargs = {attrs[0]: attr1_ft, attrs[1]: attr2_ft}
                    all_shapes.append(Obj(**kwargs))

    return np.array(all_shapes)

def make_permutations(num_permutes, num_elts):
    perms = []
    for i in range(num_permutes):
        rng = np.random.default_rng(seed=i)
        perms.append(rng.permutation(num_elts))
    return perms

with open("./src/utils/custom_permutations.pkl", 'rb') as f:
  perms = pickle.load(f)

def get_perm(perm_num: int):
  return perms[perm_num]

def make_object_list(feature_set: Type[FeatureSet], permnum: int):
    permutation = get_perm(permnum)
    return make_all_objects(feature_set)[permutation]


def paragraph_rep(num: int, 
                  objects: List[List[Obj]], 
                  labels: List[List[bool]], 
                  obj_rep_fn: Callable[Type[Obj], str]=lambda x: x.format_str(),
                  ans_rep_fn: Callable[str, str]=lambda x: x,
                  show_last=True):
    count = 0
    output = ""
    for objlist, labellist in zip(objects, labels):
        for obj, label in zip(objlist, labellist):
            count += 1
            if count == num:
                if show_last:
                    output += obj_rep_fn(obj) + ": " + ans_rep_fn(str(label))
                else: 
                    output += obj_rep_fn(obj) + ":"
                return output, ans_rep_fn(str(label))
            else:
                output += obj_rep_fn(obj) + ": " + ans_rep_fn(str(label)) + "\n"
            
    return output, ans_rep_fn(str(label))


# perm1 = [19,  4, 10, 11, 26,  2, 25,  6, 16, 23,  3, 21,  8,  0, 20, 12, 18, 13,  7,  5, 17, 14, 22,  9, 24,  1, 15]
# perm2 = [ 1,  7, 15, 23, 26, 20, 16, 25, 17, 24, 11,  2, 21, 12, 10,  3,  4, 5,  8,  0,  9, 14, 18, 22, 13,  6, 19]
