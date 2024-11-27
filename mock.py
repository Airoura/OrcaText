import json
import threading
import concurrent.futures
import numpy as np
import os
import logging
import base64
import random

from tqdm.auto import tqdm
from PIL import Image
from src.llm_backend import LLMBackend
from src.utils import *
from src.prompts import summary_prompt, profile_prompt, knowledge_prompt, deduce_prompt

#默认的warning级别，只输出warning以上的
#使用basicConfig()来指定日志级别和相关信息
logging.basicConfig(
    level=logging.DEBUG, #设置日志输出格式
    filename="logs/mock.log", #log日志输出的文件位置和文件名
    filemode="a", #文件的写入格式，w为重新写入文件，默认是追加
    format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s", #日志输出的格式
    datefmt="%Y-%m-%d %H:%M:%S", #时间输出的格式
)

max_workers = 30
MAX_RETRY_TIMES = 5

data_path = "data"
logs_path = "logs"

preview_path = os.path.join(data_path, "preview")
posts_path = os.path.join(data_path, "id_post_map")
user_info_path = os.path.join(data_path, "user_info_map")
user_posts_path = os.path.join(data_path, "user_posts_map")
user_scores_path = os.path.join(data_path, "user_avg_scores")

user_traits_path = os.path.join(data_path, "user_traits")
user_profiles_path = os.path.join(data_path, "user_profiles")
# dataset_path = os.path.join(data_path, "dataset")
deduce_queries_path = os.path.join(data_path, "deduce_queries")

user_avg_scores_path = os.path.join(preview_path, "user_avg_scores.json")
user_explanations_path = os.path.join(preview_path, "user_explanations.json")

media_path = os.path.join(data_path, "media")
media_images_path = os.path.join(media_path, "images")
media_videos_path = os.path.join(media_path, "videos")

media_images_caption_path = os.path.join(data_path, "captions")
user_conversations_path = os.path.join(data_path, "user_conversations")

post_knowledges_path = os.path.join(data_path, "knowledges")

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
llm.test()

user_avg_scores = read_user_avg_scores(user_avg_scores_path)
user_explanations = read_user_explanations(user_explanations_path)
users_info = read_users_info(user_info_path)
user_posts = read_user_posts(user_posts_path)
user_conversations = read_user_conversations(user_conversations_path)
post_ids = read_posts_id_map(posts_path)

check_dirs(user_traits_path)
check_dirs(user_profiles_path)
check_dirs(post_knowledges_path)
check_dirs(deduce_queries_path)

print(list(user_conversations.keys())[0])
print(len(list(user_explanations.keys())))

## Process Summaries
summaries = []
# 遍历 user_scores 文件夹内的文件
already_summaried = []
for filename in tqdm(os.listdir(user_traits_path)):
    # 检查文件是否为JSON文件
    if filename.endswith('.json'):
        file_path = os.path.join(user_traits_path, filename)
        # with open(file_path, 'r', encoding="utf8") as f:
        #     score_info = json.load(f)
        base_name, extension = os.path.splitext(filename)
        already_summaried.append(base_name)

print(f"already_summaried numbers:\t{len(already_summaried)}")

for user, explanation in user_explanations.items():
    userName = user.split("-", 2)[-1]
    nickName = users_info[userName]["name"]
    # if userName == "ClickHouseDB":
    #     continue
    summary_query = summary_prompt.format(user=nickName, evaluation=explanation)
    summary_dic = {
        "id": user,
        "content": summary_query
    }
    summaries.append(summary_dic)

def summary(query, phar):
    q_id = query["id"]
    if q_id in already_summaried:
        phar.update(1)
        return
    q_content = query["content"]
    try_time = 0
    while try_time < MAX_RETRY_TIMES:
        try:
            res = llm.request(q_content)
            file_path = os.path.join(user_traits_path, f"{q_id}.json")
            sj = {
                "id": q_id,
                "trait": res
            }
            save_dic2json(file_path, sj)
            # phar.update(1)
            break
        except Exception as e:
            print(e)
            file_path = os.path.join(logs_path, f"error-{try_time}-{q_id}.json")
            try_time += 1
    phar.update(1)

print(summaries[0])
print("testing...")
print(llm.request(summaries[0]["content"]))

# 多线程请求
toal_summary_num = len(summaries)
phar_summary = tqdm(total=toal_summary_num)
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    for i in summaries:
        future = executor.submit(summary, i, phar_summary)


## Process Profiles
user_traits = read_user_traits(user_traits_path)

# 遍历 user_scores 文件夹内的文件
already_profiled = []
for filename in tqdm(os.listdir(user_profiles_path)):
    # 检查文件是否为JSON文件
    if filename.endswith('.json'):
        file_path = os.path.join(user_profiles_path, filename)
        # with open(file_path, 'r', encoding="utf8") as f:
        #     score_info = json.load(f)
        base_name, extension = os.path.splitext(filename)
        already_profiled.append(base_name)

print(f"already_profiled numbers:\t{len(already_profiled)}")

profiles = []
for user, trait in user_traits.items():
    userName = user.split("-", 2)[-1]
    nickName = users_info[userName]["name"]
    conv = user_conversations[user]
    sample_conv = random.sample(conv, 10)
    profile_query = profile_prompt.format(user=nickName, posts=sample_conv)
    profile_dic = {
        "id": user,
        "content": profile_query
    }
    profiles.append(profile_dic)

print(profiles[0])

print(llm.request(profiles[0]["content"]))

def profile(query, phar):
    q_id = query["id"]
    if q_id in already_profiled:
        phar.update(1)
        return
    q_content = query["content"]
    try_time = 0
    while try_time < MAX_RETRY_TIMES:
        try:
            res = llm.request(q_content)
            file_path = os.path.join(user_profiles_path, f"{q_id}.json")
            sj = {
                "id": q_id,
                "profile": res
            }
            save_dic2json(file_path, sj)
            # phar.update(1)
            break
        except Exception as e:
            print(e)
            file_path = os.path.join(logs_path, f"error-{try_time}-{q_id}.json")
            try_time += 1
    phar.update(1)

# 多线程请求
toal_profile_num = len(profiles)
phar_profile = tqdm(total=toal_profile_num)
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    for i in profiles:
        future = executor.submit(profile, i, phar_profile)


## Potential Knowledge
user_profiles = read_user_profiles(user_profiles_path)

# 遍历 user_scores 文件夹内的文件
already_knowledged = []
for filename in tqdm(os.listdir(post_knowledges_path)):
    # 检查文件是否为JSON文件
    if filename.endswith('.json'):
        file_path = os.path.join(post_knowledges_path, filename)
        # with open(file_path, 'r', encoding="utf8") as f:
        #     score_info = json.load(f)
        base_name, extension = os.path.splitext(filename)
        already_knowledged.append(base_name)

print(f"already_knowledged numbers:\t{len(already_knowledged)}")

# 定义接收的数据格式
from pydantic import BaseModel, Field
class PotentialInfo(BaseModel):
    ContainKnowledge: bool = Field(description="Whether the content of the conversation clearly implies knowledge.")
    DetailedKnowledge: str = Field(description="Simulated the potential knowledge behind the conversation in detail.")
# 创建输出解析器
from langchain.output_parsers import PydanticOutputParser

potential_output_parser = PydanticOutputParser(pydantic_object=PotentialInfo)
# 获取输出格式指示
potential_format_instructions = potential_output_parser.get_format_instructions()
# 打印提示
print("输出格式：", potential_format_instructions)
# # 解析模型的输出
# potential_parsed_output = potential_output_parser.parse(output)
# # 将Pydantic格式转换为字典
# potential_parsed_output_dict = potential_parsed_output.dict()
# # 打印字典
# print("输出的数据：", potential_parsed_output_dict)

knowledge_queries = []
for user, conversations in user_conversations.items():
    userName = user.split("-", 2)[-1]
    user_info = users_info[userName]
    nickName = user_info["name"]

    for conversation in conversations:
        meta_data = conversation["meta_data"]
        sample_id = meta_data["conve_id"]
        conversation_str = str(conversation["conversation"]).replace("Post", "Content")
        knowledge_query = knowledge_prompt.format(
            potential_format_instructions=potential_format_instructions, 
            conversation=conversation_str
        )
        knowledge_dic = {
            "id": sample_id,
            "content": knowledge_query
        }
        knowledge_queries.append(knowledge_dic)

res = llm.request(knowledge_queries[9000]["content"])

# 解析模型的输出
potential_parsed_output = potential_output_parser.parse(res)
# 将Pydantic格式转换为字典
potential_parsed_output_dict = potential_parsed_output.dict()
# 打印字典
print("输出的数据：", potential_parsed_output_dict)

def perform_knowledge(query, phar):
    q_id = query["id"]
    if q_id in already_knowledged:
        phar.update(1)
        return
    q_content = query["content"]
    try_time = 0
    while try_time < MAX_RETRY_TIMES:
        try:
            file_path = os.path.join(post_knowledges_path, f"{q_id}.json")
            if os.path.exists(file_path):
                break
            res = llm.request(q_content)
            # 解析模型的输出
            potential_parsed_output = potential_output_parser.parse(res)
            # 将 Pydantic 格式转换为字典
            potential_parsed_output_dict = potential_parsed_output.dict()
            knowledge_result = {
                "id": q_id,
                "knowledge": potential_parsed_output_dict
            }
            save_dic2json(file_path, knowledge_result)
            break
        except Exception as e:
            print(res)
            print(e)
            try_time += 1
    phar.update(1)

# 多线程请求
toal_knowledge_num = len(knowledge_queries)
phar_knowledge = tqdm(total=toal_knowledge_num)
perform_knowledge(knowledge_queries[0], phar_knowledge)

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    for query in knowledge_queries:
        future = executor.submit(perform_knowledge, query, phar_knowledge)


## Deduce
post_knowledges = read_post_knowledges(post_knowledges_path)

# 定义接收的数据格式
from pydantic import BaseModel, Field
class DeduceInfo(BaseModel):
    rmr: bool = Field(description="Whether my Post shows the content of my profile.")
    ept: bool = Field(description="Whether my Post provides explicit evidence of my personality traits.")
    bpa: str = Field(description="A brief related psychological activities when I post my Post.")
# 创建输出解析器
from langchain.output_parsers import PydanticOutputParser

output_parser = PydanticOutputParser(pydantic_object=DeduceInfo)
# 获取输出格式指示
format_instructions = output_parser.get_format_instructions()
# 打印提示
print("输出格式：", format_instructions)
# # 解析模型的输出
# parsed_output = output_parser.parse(output)
# # 将Pydantic格式转换为字典
# parsed_output_dict = parsed_output.dict()
# # 打印字典
# print("输出的数据：", parsed_output_dict)

print(len(list(post_knowledges.keys())))

deduces = []
for user, explanation in tqdm(user_explanations.items()):
    userName = user.split("-", 2)[-1]
    nickName = users_info[userName]["name"]
    user_trait = user_traits[user]["trait"]
    user_profile = user_profiles[user]["profile"]
    if user_trait == "":
        continue
    for conversation in user_conversations[user]:
        meta_data = conversation["meta_data"]
        if meta_data["isQuote"]:
            continue
        sample_id = meta_data["conve_id"]
        try:
            knowledge = post_knowledges[sample_id]["knowledge"]
        except Exception as e:
            print(e)
            continue
        if not knowledge["ContainKnowledge"]:
            continue
        deduce_query = deduce_prompt.format(
            user=nickName, 
            profile=user_profile, 
            traits=user_trait, 
            pk=knowledge["DetailedKnowledge"], 
            conversation=conversation["conversation"],
            format_instructions=format_instructions
        )
        deduce_dic = {
            "id": sample_id,
            "fid": user,
            "uid": userName,
            "content": deduce_query
        }
        deduces.append(deduce_dic)

print(len(deduces))

content = deduces[0]["content"]
print(content)
res = llm.request(content)
print(res)

# 解析模型的输出
parsed_output = output_parser.parse(res)
# 将Pydantic格式转换为字典
parsed_output_dict = parsed_output.dict()
# 打印字典
print("输出的数据：", parsed_output_dict)

for item in tqdm(deduces):
    q_id = item["id"]
    file_path = os.path.join(deduce_queries_path, f"{q_id}.json")
    save_dic2json(file_path, item)






