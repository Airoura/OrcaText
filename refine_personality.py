import os
import re
import json
import math
import random
import decimal
import threading
import numpy as np
import pandas as pd
import concurrent.futures

from tqdm.auto import tqdm
from src.llm_backend import LLMBackend
from src.utils import *
from src.prompts import *

data_path = "data"
logs_path = "logs"

# 单个文件目录
preview_path = os.path.join(data_path, "preview")
user_avg_scores_path = os.path.join(preview_path, "user_avg_scores.json")
user_profiles_path = os.path.join(data_path, "user_profiles")
user_traits_path = os.path.join(data_path, "user_traits")
user_conversations_path = os.path.join(data_path, "user_conversations")
interpreted_summaries_path = os.path.join(data_path, "interpreted_user_summaries")

check_dirs(interpreted_summaries_path)

glm_public = LLMBackend(
    platform="zhipuai", 
    base_url="https://open.bigmodel.cn/api/paas/v4", 
    api_key="120985c00e389dac93ae62522ab5ae7a.lX6mrF4YEcSw4fmq",
    model="glm-4-flash"
)

llm = glm_public

llm.test()

user_avg_scores = read_user_avg_scores(user_avg_scores_path)

user_profiles = read_user_profiles(user_profiles_path)
user_traits = read_user_traits(user_traits_path)
user_conversations = read_user_conversations(user_conversations_path)
# 读取数据

# interpret_prompt = """
# ### Instruction
# From a professional perspective, summarize the BigFive personality traits in the assessment report.

# ### Assessment Report
# {report}

# ### Requirement
# 1. Delete the guiding sentences, such as:
#     Example 1.1 '[UserName], it's a pleasure to provide you with a summary of your Big Five personality traits based on the assessment results. \n\n';
#     Example 1.2 'Here\'s a simulated personal profile for [UserName]:\n\n'.
# 2. Delete the sentences that leaked Twitter content, such as:
#     Example 2.1 'My posts reflect my thoughts on various philosophical topics, and I'm always open to engaging in respectful and thought-provoking conversations.'.
# 3. The user name must not appear in the response.

# ### Response Example
#     The personality is characterized by a high degree of creativity and intellectual curiosity, balanced with a moderate level of conscientiousness and agreeableness. The emotional stability and resilience are notable strengths. While there may be areas for growth in terms of organization and social engagement, the overall personality profile suggests a well-rounded and adaptable individual.

# ### Response
# """

interpret_prompt = """
### Instruction
You are now an extremely intelligent and precise office assistant. 
When I give you a personality trait report and score in parentheses.
Extract corresponding information from personality trait reports to interpret BigFive and sub dimensional scores.

### Score 
Each dimension of Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism) has 6 sub dimensions (range 0-1).
The scores of the BigFive personality traits are the sum of the corresponding sub dimension scores (range 0-6).

### Openness [{o}]: 
    1. Imaginative: It shows that a person likes to be full of fantasy and create a more interesting and rich world. Imaginative and daydreaming. [{o1}]
    2. Artistic: It shows that a person values aesthetic experience and can be moved by art and beauty. [{o2}]
    3. Emotionally-aware: It shows that a person easily perceives his emotions and inner world. [{o3}]
    4. Actions: It shows that a person likes to touch new things, travel outside and experience different experiences. [{o4}]
    5. Intellectual: It shows that a person is curious, analytical, and theoretically oriented. [{o5}]
    6. Liberal: It shows that a person likes to challenge authority, conventions, and traditional ideas. [{o6}]
    
### Conscientiousness [{c}]: 
    1. Self-assured: It show that this person is confident in his own abilities. [{c1}]
    2. Organized: It shows that this person is well organized, likes to make plans and follow the rules. [{c2}]
    3. Dutiful: It shows that this person is responsible, trustworthy, polite, organized, and meticulous. [{c3}]
    4. Ambitious: It shows that this person pursues success and excellence, usually has a sense of purpose, and may even be regarded as a workaholic by others. [{c4}]
    5. Disciplined: It shows that this person will do his best to complete work and tasks, overcome difficulties, and focus on his own tasks. [{c5}]
    6. Cautious: It shows that this person is cautious, logical, and mature. [{c6}]
    
### Extraversion [{e}]:
    1. Friendly: It shows that this person often expresses positive and friendly emotions to those around him. [{e1}]
    2. Sociable: It shows that this person likes to get along with others and likes crowded occasions. [{e2}]
    3. Assertive: It show that this person likes to be in a dominant position in the crowd, directing others, and influencing others' behavior. [{e3}]
    4. Energetic: It shows that this person is energetic, fast-paced, and full of energy. [{e4}]
    5. Adventurous: It shows that this person likes noisy noise, likes adventure, seeks excitement, flashy, seeks strong excitement, and likes adventure. [{e5}]
    6. Cheerful: It shows that this person easily feels various positive emotions, such as happiness, optimism, excitement, etc. [{e6}]
    
### Agreeableness [{a}]:
    1. Trusting: It show that the person believes that others are honest, credible, and well-motivated. [{a1}]
    2. Genuine: It show that the person thinks that there is no need to cover up when interacting with others, and appear frank and sincere. [{a2}]
    3. Generous: It show that this person is willing to help others and feel that helping others is a pleasure. [{a3}]
    4. Compliance: It show that this person does not like conflicts with others, in order to get along with others, willing to give up their position or deny their own needs. [{a4}]
    5. Humblel: It shows that this person does not like to be pushy and unassuming. [{a5}]
    6. Empathetic: It show that the person is compassionate and easy to feel the sadness of others. [{a6}] 
    
### Neuroticism [{n}]:
    1. Anxiety-prone: It shows that this person is easy to feel danger and threat, easy to be nervous, fearful, worried, and upset. [{n1}]
    2. Aggressive: It shows that this person is easy to get angry, and will be full of resentment, irritability, anger and frustration after feeling that he has been treated unfairly. [{n2}]
    3. Melancholy: It shows that this person is easy to feel sad, abandoned, and discouraged. [{n3}]
    4. Self-conscious: It shows that this person is too concerned about how others think of themselves, is afraid that others will laugh at themselves, and tend to feel shy, anxious, low self-esteem, and embarrassment in social situations. [{n4}]
    5. Impulsive: It shows that when the person feels strong temptation, it is not easy to restrain, and it is easy to pursue short-term satisfaction without considering the long-term consequences. [{n5}]
    6. Stress-prone: It shows that this person has poor ability to cope with stress, becoming dependent, losing hope, and panicking when encountering an emergency. [{n6}]

### Report
{report}

### Requirement
1. Delete the guiding sentences, such as: 'Here is the response based on the provided personality trait report: \n\n'.
2. Delete the sentences that leaked Twitter content.
3. The username and behavior must not appear in the response.

### Response Example
    Openness: High (3.69) - intellectually curious, open-minded, and enjoys exploring complex ideas. High scores in Imaginative, Emotionally-aware, Intellectual, and Liberal sub-dimensions.\n\nConscientiousness: Low (1.0) - not particularly organized or disciplined, with a relaxed approach to life. Only high score in Self-assured sub-dimension.\n\nExtraversion: Low (1.23) - assertive and sociable, but not overly friendly or cheerful. High score in Assertive sub-dimension.\n\nAgreeableness: Low (0.54) - values intellectual honesty and critical thinking over interpersonal harmony. Only high score in Genuine sub-dimension.\n\nNeuroticism: Low (0.0) - no notable signs of anxiety or stress, with low scores across all sub-dimensions.


### Response
"""

# interpret_prompt = """
# ### Instruction
# Assuming you are a seasoned psychologist, interprete the personality trait score in parentheses.

# ### Score 
# Each dimension of Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism) has 6 sub dimensions (range 0-1).
# The scores of the BigFive personality traits are the sum of the corresponding sub dimension scores (range 0-6).

# ### Openness [{o}]: 
#     1. Imaginative: It shows that a person likes to be full of fantasy and create a more interesting and rich world. Imaginative and daydreaming. [{o1}]
#     2. Artistic: It shows that a person values aesthetic experience and can be moved by art and beauty. [{o2}]
#     3. Emotionally-aware: It shows that a person easily perceives his emotions and inner world. [{o3}]
#     4. Actions: It shows that a person likes to touch new things, travel outside and experience different experiences. [{o4}]
#     5. Intellectual: It shows that a person is curious, analytical, and theoretically oriented. [{o5}]
#     6. Liberal: It shows that a person likes to challenge authority, conventions, and traditional ideas. [{o6}]
    
# ### Conscientiousness [{c}]: 
#     1. Self-assured: It show that this person is confident in his own abilities. [{c1}]
#     2. Organized: It shows that this person is well organized, likes to make plans and follow the rules. [{c2}]
#     3. Dutiful: It shows that this person is responsible, trustworthy, polite, organized, and meticulous. [{c3}]
#     4. Ambitious: It shows that this person pursues success and excellence, usually has a sense of purpose, and may even be regarded as a workaholic by others. [{c4}]
#     5. Disciplined: It shows that this person will do his best to complete work and tasks, overcome difficulties, and focus on his own tasks. [{c5}]
#     6. Cautious: It shows that this person is cautious, logical, and mature. [{c6}]
    
# ### Extraversion [{e}]:
#     1. Friendly: It shows that this person often expresses positive and friendly emotions to those around him. [{e1}]
#     2. Sociable: It shows that this person likes to get along with others and likes crowded occasions. [{e2}]
#     3. Assertive: It show that this person likes to be in a dominant position in the crowd, directing others, and influencing others' behavior. [{e3}]
#     4. Energetic: It shows that this person is energetic, fast-paced, and full of energy. [{e4}]
#     5. Adventurous: It shows that this person likes noisy noise, likes adventure, seeks excitement, flashy, seeks strong excitement, and likes adventure. [{e5}]
#     6. Cheerful: It shows that this person easily feels various positive emotions, such as happiness, optimism, excitement, etc. [{e6}]
    
# ### Agreeableness [{a}]:
#     1. Trusting: It show that the person believes that others are honest, credible, and well-motivated. [{a1}]
#     2. Genuine: It show that the person thinks that there is no need to cover up when interacting with others, and appear frank and sincere. [{a2}]
#     3. Generous: It show that this person is willing to help others and feel that helping others is a pleasure. [{a3}]
#     4. Compliance: It show that this person does not like conflicts with others, in order to get along with others, willing to give up their position or deny their own needs. [{a4}]
#     5. Humblel: It shows that this person does not like to be pushy and unassuming. [{a5}]
#     6. Empathetic: It show that the person is compassionate and easy to feel the sadness of others. [{a6}] 
    
# ### Neuroticism [{n}]:
#     1. Anxiety-prone: It shows that this person is easy to feel danger and threat, easy to be nervous, fearful, worried, and upset. [{n1}]
#     2. Aggressive: It shows that this person is easy to get angry, and will be full of resentment, irritability, anger and frustration after feeling that he has been treated unfairly. [{n2}]
#     3. Melancholy: It shows that this person is easy to feel sad, abandoned, and discouraged. [{n3}]
#     4. Self-conscious: It shows that this person is too concerned about how others think of themselves, is afraid that others will laugh at themselves, and tend to feel shy, anxious, low self-esteem, and embarrassment in social situations. [{n4}]
#     5. Impulsive: It shows that when the person feels strong temptation, it is not easy to restrain, and it is easy to pursue short-term satisfaction without considering the long-term consequences. [{n5}]
#     6. Stress-prone: It shows that this person has poor ability to cope with stress, becoming dependent, losing hope, and panicking when encountering an emergency. [{n6}]

# ### Requirement
# 1. Delete the guiding sentences, such as: 'Here is the response based on the provided personality trait report: \n\n'.
# 2. Delete the sentences that leaked Twitter content.


# ### Response
# """

import math
def round_number(scores):
    results = []
    for i in scores:
        result = round(i, 2)
        results.append(result)
    return results

interpret_queries = []
datasets = []
final_users = []
for user, conversations in tqdm(user_conversations.items()):
    # if user not in real_selected_original_users:
    #     continue
    final_users.append(user)
    if user not in user_avg_scores:
        continue
    user_avg_score = round_number(user_avg_scores[user])
    trait = user_traits[user]["trait"]
    query = interpret_prompt.format(
        report=trait,
        o=user_avg_score[0],
        o1=user_avg_score[1],
        o2=user_avg_score[2],
        o3=user_avg_score[3],
        o4=user_avg_score[4],
        o5=user_avg_score[5],
        o6=user_avg_score[6],
        c=user_avg_score[7],
        c1=user_avg_score[8],
        c2=user_avg_score[9],
        c3=user_avg_score[10],
        c4=user_avg_score[11],
        c5=user_avg_score[12],
        c6=user_avg_score[13],
        e=user_avg_score[14],
        e1=user_avg_score[15],
        e2=user_avg_score[16],
        e3=user_avg_score[17],
        e4=user_avg_score[18],
        e5=user_avg_score[19],
        e6=user_avg_score[20],
        a=user_avg_score[21],
        a1=user_avg_score[22],
        a2=user_avg_score[23],
        a3=user_avg_score[24],
        a4=user_avg_score[25],
        a5=user_avg_score[26],
        a6=user_avg_score[27],
        n=user_avg_score[28],
        n1=user_avg_score[29],
        n2=user_avg_score[30],
        n3=user_avg_score[31],
        n4=user_avg_score[32],
        n5=user_avg_score[33],
        n6=user_avg_score[34],
    )
    dic = {
        "id": user,
        "content": query,
    }
    interpret_queries.append(dic)

print(len(interpret_queries))

print(interpret_queries[0])

res = llm.request(interpret_queries[0]["content"])

print(res.replace("*", ""))

# file_path = os.path.join(interpreted_resumes_path, f"Alice.json")

# save_dic2json(file_path, sj)

def interpret(query, phar):
    q_id = query["id"]
    q_content = query["content"]
    try_time = 0
    while try_time < 5:
        try:
            file_path = os.path.join(interpreted_summaries_path, f"{q_id}.json")
            if os.path.exists(file_path):
                break
            res = llm.request(q_content)
            # 将Pydantic格式转换为字典
            sj = {
                "result": res.replace("*", "")
            }
            save_dic2json(file_path, sj)
            # phar.update(1)
            break
        except Exception as e:
            print(e)
            try_time += 1
    phar.update(1)

# interpret(interpret_queries[0], phar_desensitize)

toal_interpret_num = len(interpret_queries)
phar_interpret = tqdm(total=toal_interpret_num)
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    for i in interpret_queries:
        future = executor.submit(interpret, i, phar_interpret)

## Already interpreted

# 遍历 user_scores 文件夹内的文件
interpreted_summaries = {}
for filename in tqdm(os.listdir(interpreted_summaries_path)):
    # 检查文件是否为JSON文件
    if filename.endswith('.json'):
        file_path = os.path.join(interpreted_summaries_path, filename)
        with open(file_path, 'r', encoding="utf8") as f:
            interpreted_summary = json.load(f)
        base_name, extension = os.path.splitext(filename)
        interpreted_summaries[base_name] = interpreted_summary

print(list(interpreted_summaries.values())[0])

# peit_pretrain_prompt = """
# The personality score is {score}. From a professional perspective, summarize the BigFive personality traits.
# """

# peit_pretrain_prompt = """{score}."""

PERSONALITY_TOKEN = "<|reserved_special_token_0|>"
personality_embedding_tokens = PERSONALITY_TOKEN * 5

peit_pretrain_prompt_simple = """Assuming you are a seasoned psychologist, you are evaluating the degree of BigFive in the personality scores: {score}, categorize the degree of each personality traits into high or low."""

peit_pretrain_prompt_complex = """Assuming you are a seasoned psychologist, interprete the BigFive personality scores: {score}."""

degree_res = """Openness: {openness}, Conscientiousness: {conscientiousness}, Extraversion: {extraversion}, Agreeableness: {agreeableness}, Neuroticism: {neuroticism}"""

simple_mode = False

pre_datasets = []
for user, conversations in tqdm(user_conversations.items()):
    # if user in real_selected_original_users:
    #     continue
    if user not in user_avg_scores.keys():
        continue
    user_avg_score = round_number(user_avg_scores[user])
    if simple_mode:
        ins = peit_pretrain_prompt_simple.format(
            score=personality_embedding_tokens,
        )
        if user_avg_score[0] > 3:
            o = f"high ({user_avg_score[0]})"
        else:
            o = f"low ({user_avg_score[0]})"
        if user_avg_score[7] > 3:
            c = f"high ({user_avg_score[7]})"
        else:
            c = f"low ({user_avg_score[7]})"
        if user_avg_score[14] > 3:
            e = f"high ({user_avg_score[14]})"
        else:
            e = f"low ({user_avg_score[14]})"
        if user_avg_score[21] > 3:
            a = f"high ({user_avg_score[21]})"
        else:
            a = f"low ({user_avg_score[21]})"
        if user_avg_score[28] > 3:
            n = f"high ({user_avg_score[28]})"
        else:
            n = f"low ({user_avg_score[28]})"
    
        opt = degree_res.format(
            openness=o,
            conscientiousness=c,
            extraversion=e,
            agreeableness=a,
            neuroticism=n
        )
    else:
        ins = peit_pretrain_prompt_complex.format(
            score=personality_embedding_tokens,
        )
        opt = interpreted_summaries[user]["result"]
    pre_dic = {
        "meta_data": {
            "user": user,
            "score": user_avg_score
        },
        "instruction": ins,
        "output": opt
    }
    pre_datasets.append(pre_dic)

pre_datasets[0]

print(len(pre_datasets))

pre_datasets_complex = pre_datasets

pre_datasets = pre_datasets + pre_datasets_complex

random.shuffle(pre_datasets)

pre_file_path = os.path.join(data_path, "finetunes", "pcet_pretrain.json")
save_dic2json(pre_file_path, pre_datasets)









