import json
import pandas as pd
import math
import decimal
import concurrent.futures
import threading
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import base64
import random

from tqdm.auto import tqdm
from pprint import pprint
from src.llm_backend import LLMBackend
from src.utils import *
from src.prompts import bigfive_score_criteria, system_prompt

max_workers = 30
convs_per_chunk = 10
MAX_RETRY_TIMES = 5

data_path = "data"
logs_path = "logs"

posts_path = os.path.join(data_path, "id_post_map")
user_info_path = os.path.join(data_path, "user_info_map")
user_posts_path = os.path.join(data_path, "user_posts_map")
user_scores_path = os.path.join(data_path, "user_scores")
user_conversations_path = os.path.join(data_path, "user_conversations")

check_dirs(user_scores_path)

glm_public = LLMBackend(
    platform="zhipuai", 
    base_url="https://open.bigmodel.cn/api/paas/v4", 
    api_key="120985c00e389dac93ae62522ab5ae7a.lX6mrF4YEcSw4fmq",
    model="glm-4-flash"
)
# llama = LLMBackend(
#     platform="openai", 
#     base_url="http://172.16.64.188:8000/v1", 
#     api_key="-",
#     model="llama3.1-70b"
# )

llm = glm_public

llm.test("hi")

users_info = read_users_info(user_info_path)

# 遍历 user_scores 文件夹内的文件
already_scored = []
for filename in tqdm(os.listdir(user_scores_path)):
    # 检查文件是否为JSON文件
    if filename.endswith('.json'):
        file_path = os.path.join(user_scores_path, filename)
        with open(file_path, 'r', encoding="utf8") as f:
            score_info = json.load(f)
        base_name, extension = os.path.splitext(filename)
        already_scored.append(base_name)

print(f"already_scored numbers:\t{len(already_scored)}")

user_conversations = read_user_conversations(user_conversations_path)

print(user_conversations["enfj-history-DrJEBall"])

chunks = []
for user, convs in tqdm(user_conversations.items()):
    chunk_id = 0
    convs_len = len(convs)
    chunks_num = math.ceil(convs_len / convs_per_chunk)
    while chunk_id < chunks_num:
        start_index = chunk_id * convs_per_chunk
        if chunk_id == chunks_num - 1:
            end_index = -1
        else:
            end_index = (chunk_id + 1) * convs_per_chunk
        # chunk_conv = convs[start_index: end_index]
        chunk_conv = []
        for item in convs[start_index: end_index]:
            chunk_conv.append(item["conversation"])
        chunk_id_str = f"{user}-{chunk_id}"
        dic = {
            "id": chunk_id_str,
            "content": chunk_conv
        }
        chunks.append(dic)
        chunk_id += 1

print(len(chunks))

print(chunks[0])

queries = []
for chunk in tqdm(chunks):
    conv_id = chunk["id"]
    if conv_id in already_scored:
        continue
    conv = chunk["content"]
    user_name = conv_id.split("-", 3)[2]
    user_info = users_info[user_name]
    user_dic = {
        "userName": user_name,
        "nickName": user_info["name"],
        "description": user_info["description"],
        "followers": user_info["followers"],
        "following": user_info["following"],
        "favouritesCount": user_info["favouritesCount"],
        "mediaCount": user_info["mediaCount"]
    }
    name = user_dic["nickName"]
    query = system_prompt.format(criteria=bigfive_score_criteria, user=user_dic, name=name, conversation=conv)
    query_dic = {
        "id": conv_id,
        "content": query
    }
    queries.append(query_dic)

if len(queries) == 0:
    exit(0)
    
print(queries[0])

def score(query, phar):
    q_id = query["id"]
    q_content = query["content"]

    try_time = 0
    while try_time < MAX_RETRY_TIMES:
        try:
            file_path = os.path.join(user_scores_path, f"{q_id}.json")
            if os.path.exists(file_path):
                break
            re = llm.request(q_content)
            sj = load_json(re)
            save_dic2json(file_path, sj)
            break
        except Exception as e:
            print(e)
            try_time += 1
    phar.update(1)

print("testing llm request...")
res = llm.request(queries[0]["content"])

print(load_json(res))

# 多线程请求
toal = len(queries)
phar = tqdm(total=toal)

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    for i in queries:
        future = executor.submit(score, i, phar)











