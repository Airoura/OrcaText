import json
import threading
import concurrent.futures
import numpy as np
import os
import logging
import base64

from tqdm.auto import tqdm
from PIL import Image
from src.llm_backend import LLMBackend
from src.utils import *

data_path = "data"
logs_path = "logs"

preview_path = os.path.join(data_path, "preview")
user_posts_path = os.path.join(data_path, "user_posts_map")
user_posts = read_user_posts(user_posts_path)


# 把用户推特中出现的图片地址保存下来供 caption_image.py caption。

medias = []
extentions = []
failed_count = 0
for user, posts in tqdm(user_posts.items()):
    if failed_count >= 10:
        break
    for post in posts:
        if post:
            media = post["media"]
            for m in media:
                dic = {
                    "caption": m,
                    "url": m
                }
                if "https://pbs.twimg.com" not in m:
                    print(m)
                # assert(base_name == "https://pbs.twimg.com/media")
                medias.append(dic)
                extentions.append(m.split(".")[-1])
                
            if "quote" in post:
                quote = post["quote"]
                try:
                    if "entities" in quote:
                        entities = quote["entities"]
                    elif "extendedEntities" in quote:
                        entities = quote["extendedEntities"]
                    else:
                        continue
                    if "media" in entities:
                        media = entities["media"]
                        for m in media:
                            dic = {
                                "caption": m,
                                "url": m["media_url_https"]
                            }
                            # if "https://pbs.twimg.com" not in m["media_url_https"]:
                            #     print(m)
                            # assert(base_name == "https://pbs.twimg.com/media")
                            medias.append(dic)
                            extentions.append(m["media_url_https"].split(".")[-1])
                except Exception as e:
                    print(e)
                    # print(post["quote"])
                    failed_count += 1
                    if failed_count >= 10:
                        break
file_path = os.path.join(preview_path, "images.json")
save_dic2json(file_path, medias)