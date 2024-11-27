import os
import json
import threading
import numpy as np
import concurrent.futures

from tqdm.auto import tqdm
from src.utils import *

data_path = "data"
logs_path = "logs"

user_info_path = os.path.join(data_path, "user_info_map")
user_scores_path = os.path.join(data_path, "user_scores")
preview_path = os.path.join(data_path, "preview")

users_info = read_users_info(user_info_path)

user_scores = read_user_scores(user_scores_path)

def add_arrays(arr1, arr2):
    return np.add(arr1, arr2).tolist()

user_average_scores = {}
user_sum_explanations = {}

current_sample = list(user_scores.keys())[0]
current_user = "-".join(current_sample.split("-")[:-1])
avg_scores = [0] * 35
explanations = []
chunk_num = 0
i = 0
total_chunk_num =  len(user_scores.keys())
for user, score in tqdm(user_scores.items()):
    # print(user)
    # break
    user_name = "-".join(user.split("-")[:-1])
    if user_name[-1] == "-":
        print(user_name)
        break
    chunk_id = user[-1]
    if user_name != current_user:
        # print(chunk_num)
        avg_scores = [x / chunk_num for x in avg_scores]
        user_average_scores[current_user] = avg_scores
        user_sum_explanations[current_user] = explanations
        avg_scores = [0] * 35
        current_user = user_name
        chunk_num = 0
        avg_score = [0] * 35
        explanations = []
        
    guard = 0
    avg_score = [0] * 35
    for bigfive, sub_dimention in score.items():
        if bigfive == "Explanation":
            explanations.append(sub_dimention)
            continue
        sub_scores = list(sub_dimention.values())
        if len(sub_scores) != 6:
            print("miss some sub-dimention scores")
            guard += 1
            continue
        bigfive_score = sum(sub_scores)
        # print(bigfive_score)
        avg_score[guard*7] = bigfive_score
        avg_score[guard*7+1:guard*7+7] = sub_scores
        
        guard += 1
    
    # print(guard)
    # print(avg_score)
    # print(len(avg_score))
    avg_scores = add_arrays(avg_scores, avg_score)
    chunk_num += 1

    i += 1
    if i == total_chunk_num:
        avg_scores = [x / chunk_num for x in avg_scores]
        user_average_scores[current_user] = avg_scores
        user_sum_explanations[current_user] = explanations

print(len(user_average_scores.keys()))

print(list(user_average_scores.values())[0])

print(total_chunk_num)

print(len(list(user_sum_explanations.keys())))

user_avg_scores_path = os.path.join(preview_path, "user_avg_scores.json")

save_dic2json(user_avg_scores_path, user_average_scores)

user_explanations_path = os.path.join(preview_path, "user_explanations.json")

save_dic2json(user_explanations_path, user_sum_explanations)

