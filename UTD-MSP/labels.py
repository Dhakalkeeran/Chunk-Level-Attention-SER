import logging
import os
import re
from glob import glob
from math import ceil
from random import shuffle

import pandas as pd

def get_labels_from_file(filepath):
    line_info = []
    with open(filepath, "r") as file:
        for idx, line in enumerate(file):
            if not line.strip().startswith("["):
                continue
            
            match = re.search(r"Ses[A-Za-z\d\_]{3,}", line)
            if match:
                filename = match.group(0)
            else:
                logging.error(f"Could not extract filename on line {idx} from file {filepath}")
            
            match = re.search(r"\_(M|F)\d+", filename)
            if match:
                gender = match.group(1)
            else:
                logging.error(f"Could not extract gender on line {idx} from file {filepath}")

            match = re.search(r"\s(xxx|neu|fru|ang|sad|hap|sur|exc|fea|dis|[a-z]{3})\s", line)
            if match:
                emo_class = match.group(1)
            else:
                logging.error(f"Could not extract emo_class on line {idx} from file {filepath}")

            match = re.search(r"\[([\d\.]+)\s?\,\s?([\d\.]+)\s?\,\s?([\d\.]+)\]", line)
            if match:
                val, act, dom = match.group(1), match.group(2), match.group(3)
            else:
                logging.error(f"Could not extract val/act/dom on line {idx} from file {filepath}")

            speaker_id = "Unknown"
            line_info.append([filename, emo_class, act, val, dom, speaker_id, gender])

    return line_info

def get_labels(dir_paths):
    file_info = []
    if isinstance(dir_paths, str):
        dir_paths = [dir_paths]
    for dir_path in dir_paths:
        dir_path = dir_path.rstrip("/")
        for filepath in glob(f"{dir_path}/*.txt"):
            file_info.extend(get_labels_from_file(filepath))
    return file_info

def find_dirs(root_path):
    dir_paths = []
    for path in glob(root_path + "/*"):
        path_name = os.path.split(path)[-1]
        if re.match("Emo.*", path_name):
            # dir_paths.append(path + "/dialog/EmoEvaluation/")
            dir_paths.append(path)
    return dir_paths

# filepath = "EmoEvaluation/Ses01F_impro01.txt"
# line_info = get_labels_from_file(filepath)

root_path = "./"
dir_paths = find_dirs(root_path)
file_info = get_labels(dir_paths)
# file_info = get_labels("EmoEvaluation/")

labels_df = pd.DataFrame(file_info, columns=["FileName", "EmoClass", "EmoAct", "EmoVal", "EmoDom", "SpkrID", "Gender"])
train_ratio = 0.6
train_split = ceil(train_ratio * len(labels_df))
test_split = ceil((len(labels_df) - train_split)/2)
val_split = len(labels_df) - train_split - test_split
split_set = ["Train" for i in range(train_split)] + ["Test" for i in range(test_split)] + ["Validation" for i in range(val_split)]

shuffle(split_set)
labels_df["Split_Set"] = split_set
print(labels_df)
labels_df.to_csv("labels_concensus.csv", index=False)





