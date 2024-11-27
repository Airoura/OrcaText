import os
import re
import json
import math
import random
import decimal
import argparse
import threading
import numpy as np
import pandas as pd
import concurrent.futures

from tqdm.auto import tqdm
from src.utils import *

data_path = "data"
logs_path = "logs"


# 单个文件目录
preview_path = os.path.join(data_path, "preview")
user_info_path = os.path.join(data_path, "user_info_map")
user_avg_scores_path = os.path.join(preview_path, "user_avg_scores.json")
deduces_path = os.path.join(data_path, "deduces")
user_profiles_path = os.path.join(data_path, "user_profiles")
user_traits_path = os.path.join(data_path, "user_traits")
user_conversations_path = os.path.join(data_path, "user_conversations")
potential_knowledges_path = os.path.join(data_path, "knowledges")
finetune_path = os.path.join(data_path, "finetunes")
social_bench_path = os.path.join(data_path, "social_bench")
real_selected_original_users_path = os.path.join(social_bench_path, "real_selected_users.json")
interpreted_summaries_path = os.path.join(data_path, "interpreted_user_summaries")

check_dirs(finetune_path)
check_dirs(social_bench_path)

real_selected_original_users = read_real_selected_original_users(real_selected_original_users_path)

real_selected_original_users

save_dic2json(real_selected_original_users_path, real_selected_original_users)

users_info = read_users_info(user_info_path)

user_profiles = read_user_profiles(user_profiles_path)

user_traits = read_user_traits(user_traits_path)

user_conversations = read_user_conversations(user_conversations_path)

potential_knowledges = read_post_knowledges(potential_knowledges_path)

user_avg_scores = read_user_avg_scores(user_avg_scores_path)

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

# Process Dataset

# 遍历文件夹内的文件
deduces = {}
deduces_ids = []
for filename in tqdm(os.listdir(deduces_path)):
    # 检查文件是否为JSON文件
    if filename.endswith('.json'):
        file_path = os.path.join(deduces_path, filename)
        with open(file_path, 'r', encoding="utf8") as f:
            deduce = json.load(f)
        base_name, extension = os.path.splitext(filename)
        deduces_ids.append(base_name)
        deduces[base_name] = deduce

print(len(set(deduces_ids)))

print(f"already_duduced numbers:\t{len(deduces_ids)}")

print(list(deduces.values())[1])

print(list(user_traits.values())[1])

PERSONALITY_TOKEN = "<|reserved_special_token_0|>"
personality_embedding_tokens = PERSONALITY_TOKEN * 5

def main(args):
    from src.prompts import write_instruction, write_instruction_without_psychology, write_instruction_without_psychology_media, reply_instruction, reply_instruction_without_psychology, reply_instruction_without_psychology_media
    failed_ids = []
    failed_samples = []
    
    pre_datasets = []
    datasets = []
    user_count = 0
    quote_count = 0
    media_count = 0
    posts_count_map = {}
    upstream_length_arr = []
    label_length_arr = []
    
    for user, conversations in tqdm(user_conversations.items()):
        if not args.select_users:
            if args.social_bench:
                if user not in real_selected_original_users:
                    continue
            else:
                if user in real_selected_original_users:
                    continue
        if user not in user_avg_scores.keys():
            continue
        user_avg_score = user_avg_scores[user]
        user_count += 1
        posts_count_map[user] = 0
        userName = user.split("-", 2)[2]
        user_info = users_info[userName]
        nick_name = user_info["name"]
        for conversation in conversations:
            meta_data = conversation["meta_data"]
            conv_id = meta_data["conve_id"]
            if conv_id not in deduces_ids:
                continue
            knowledge = potential_knowledges[conv_id]["knowledge"]
            if not knowledge["ContainKnowledge"]:
                continue
            conv = conversation["conversation"]
            for i in conv:
                if i["Media"]:
                    media_count += 1
            sample = deduces[conv_id]
    
            profile = ""
            personality = ""
            
            if sample["rmr"]:
                profile = user_profiles[user]["profile"]
            if sample["ept"]:
                if args.psit_mode:
                    # personality = personality_embedding_tokens
                    personality = interpreted_summaries[user]["result"]
                else:
                    personality = user_traits[user]["trait"]
            
            potential_knowledge = knowledge["DetailedKnowledge"]
            
            # Ablation Study
            if args.profile_ablation:
                resume = ""
            if args.personality_ablation:
                personality = ""
            if args.potential_ablation:
                potential_knowledge = ""
            
            filterd_post = post_cleaner(conv[-1]["Post"])
            # filterd_post = replace_url(replace_alt(conv[-1]["Post"])).strip()
            # filterd_post = re.sub(r'\s+', ' ', filterd_post)
            
            label_length = len(filterd_post)
            label_length_arr.append(label_length)
            
            label_dic = {
                "Psychological Activities": sample["bpa"],
                "Post Content": filterd_post,
                "Media": conv[-1]["Media"]
            }
            
            if args.without_psychology:
                write_instruction = write_instruction_without_psychology
                reply_instruction = reply_instruction_without_psychology
                label_dic = {
                    "Post Content": filterd_post,
                    "Media": conv[-1]["Media"]
                }
                
            if args.without_psychology_media:
                write_instruction = write_instruction_without_psychology_media
                reply_instruction = reply_instruction_without_psychology_media
                label_dic = {
                    "Post Content": filterd_post
                }
                
            is_reply = False
            
            if len(conv) == 1:
                inp = write_instruction.format(
                    user=nick_name,
                    personality=personality,
                    profile=profile,
                    pk=potential_knowledge,
                )
            else:
                is_reply = True
                upstream_length = len(conv[-2]["Content"])
                upstream_length_arr.append(upstream_length)
                inp = reply_instruction.format(
                    user=nick_name,
                    poster=conv[-2]["User"],
                    personality=personality,
                    profile=profile,
                    pk=potential_knowledge,
                    ut=post_cleaner(json.dumps(conv[:-1]))
                )
            dic = {
                "meta_data": {
                    "id": sample["id"],
                    "user": user,
                    "is_reply": is_reply,
                    "profile_related": sample["rmr"],
                    "personality_related": sample["ept"],
                    "score": user_avg_score
                },
                "instruction": inp,
                "output": json.dumps(label_dic)
            }
            if dic["instruction"] and dic["output"]:
                datasets.append(dic)
                posts_count_map[user] += 1
    
    average_upstream_length = np.mean(np.array(upstream_length_arr))
    average_label_length = np.mean(np.array(label_length_arr))
    
    print(f"failed_samples:\t{failed_samples}")
    
    print(f"failed_ids:\t{failed_ids}")
    
    random.shuffle(datasets)
    
    print(datasets[0]["meta_data"])
    
    print(len(datasets))
    
    is_reply_count = 0
    profile_related_count = 0
    personality_related_count = 0
    
    tagged_users = []
    for item in tqdm(datasets):
        meta_data = item["meta_data"]
        if meta_data["is_reply"]:
            is_reply_count += 1
        if meta_data["profile_related"]:
            profile_related_count += 1
        if meta_data["personality_related"]:
            personality_related_count += 1
        user = meta_data["user"]
        # 随机抽取用户
        if args.select_users and posts_count_map[user] >= 100:
            tagged_users.append(user)
    
    if args.select_users:
        unique_users = list(set(tagged_users))[:25]
        print(len(unique_users))
        # 使用with语句确保文件正确关闭
        with open(real_selected_original_users_path, 'w') as file:
            # 将列表写入到文件中
            json.dump(unique_users, file)
        exit(0)
        
    statistic_dic = {
        "user_count": user_count,
        "post_count": len(datasets),
        "quote_count": quote_count,
        "media_count": media_count,
        "is_reply_count": is_reply_count,
        "profile_related_count": profile_related_count,
        "personality_related_count": personality_related_count,
        "average_upstream_length": average_upstream_length,
        "average_label_length": average_label_length
    }
    
    print(statistic_dic)
    
    print(datasets[0]["meta_data"])
    print(datasets[0]["instruction"])
    print(datasets[0]["output"])
    
    if args.social_bench or args.select_users:
        save_dir = "social_bench"
    else:
        save_dir = "finetunes"
        
    if args.psit_mode:
        prefix = "psit"
    else:
        prefix = "ptit"
        
    if args.without_psychology:
        postfix = "without_psychology"
    elif args.without_psychology_media:
        postfix = "without_psychology_media"
    elif args.profile_ablation:
        postfix = "profile_ablation"
    elif args.personality_ablation:
        postfix = "personality_ablation"
    elif args.potential_ablation:
        postfix = "potential_ablation"
    else:
        postfix = "propose"
        
    file_path = os.path.join(data_path, save_dir, f"{prefix}_{postfix}.json")
    print(file_path)
    
    original_filename = os.path.splitext(os.path.basename(file_path))[0]
    
    save_dic2json(file_path, datasets)
    
    fourth = len(datasets) // 4
    
    fourth_path = original_filename + "_fourth" + ".json"
    
    fourth_path = os.path.join(data_path, save_dir, fourth_path)
    
    save_dic2json(fourth_path, datasets[:fourth])
    
    eighth = len(datasets) // 8
    
    eighth_path = original_filename + "_eighth" + ".json"
    
    eighth_path = os.path.join(data_path, save_dir, eighth_path)
    
    save_dic2json(eighth_path, datasets[:eighth])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OrcaText Post Dataset')
    parser.add_argument('-su', '--select_users', action='store_true', help='', default=False)
    parser.add_argument('-sb', '--social_bench', action='store_true', help='', default=False)
    parser.add_argument('-wp', '--without_psychology', action='store_true', help='', default=False)
    parser.add_argument('-wpm', '--without_psychology_media', action='store_true', help='', default=False)
    parser.add_argument('-pm', '--psit_mode', action='store_true', help='', default=False)
    parser.add_argument('-cpa', '--profile_ablation', action='store_true', help='', default=False)
    parser.add_argument('-pta', '--personality_ablation', action='store_true', help='', default=False)
    parser.add_argument('-pka', '--potential_ablation', action='store_true', help='', default=False)
    
    args = parser.parse_args()
    main(args)











