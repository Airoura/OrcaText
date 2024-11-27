import os  
import shutil  
import json
import uuid
import base64
import time
import concurrent.futures
import threading

from openai import OpenAI
from tqdm import tqdm

api_model = "THUDM/cogvlm2-llama3-chat-19B-tgi"
api_key="-"
base_url = "http://172.16.64.69:8080/v1"
max_workers = 20
convs_per_chunk = 20

data_path = "data"

preview_path = os.path.join(data_path, "preview")
media_images_config_path = os.path.join(preview_path, "images.json")
media_images_caption_path = os.path.join(data_path, "captions")
media_path = os.path.join(data_path, "media")
media_images_file_path = os.path.join(media_path, "images")
media_videos_ile_path = os.path.join(media_path, "videos")

assert os.path.exists(data_path)
assert os.path.exists(preview_path)
assert os.path.exists(media_images_config_path)

check_dirs(media_images_file_path)
check_dirs(media_images_caption_path)

MAX_RETRY_TIMES = 5

# init the client but point it to TGI
client = OpenAI(base_url=base_url, api_key=api_key)

def visual_requests(query, image, api_model):
    failed_times = 0
    response = None
    message = ""
    while True:
        try:
            chat_completion = client.chat.completions.create(
                model=api_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image
                                },
                            },
                        ],
                    },
                ],
                # max_tokens=2048,
                temperature=0.01,
                top_p=0.7,
                stream=False,
            )
            message = chat_completion.choices[0].message.content
            return message
        except Exception as e:
            print(e)
            failed_times += 1
            if failed_times >= MAX_RETRY_TIMES:
                return message

def rc_image(image_path):
    with open(image_path, "rb") as f:
        image = base64.b64encode(f.read()).decode("utf-8")
    image = f"data:image/png;base64,{image}"
    return image

query = "What's in this image?"

# 处理爬取的图片
crawl_images_file_path = "data/media/images_crawl"
threads = ["00000", "00001", "00002", "00003", "00004"]
for thread in threads:
    directory = os.path.join(crawl_images_file_path, thread)
    raw_images = []
    # 遍历文件夹内的文件
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(directory, filename)
            base_name, _ = os.path.splitext(filename)
            caption_file = os.path.join(directory, base_name + ".json")
            with open(caption_file, 'r', encoding="utf8") as f:
                meta_info = json.load(f)
                caption = meta_info["url"]
            _, extention = os.path.splitext(caption)
            encrypted_text = encode_url(caption)
            dest_path = os.path.join(media_images_file_path, encrypted_text + extention)
            shutil.copy(image_path, dest_path)

# 读取一共需要爬取的图片
with open(media_images_config_path, 'r', encoding="utf8") as f:
    media_images_config = json.load(f)

base64_images = []
failed_images = []
i = 0

for sample in tqdm(media_images_config):
    url = sample["url"]
    base_name, extention = os.path.splitext(url)
    url_decrypted = encode_url(url)
    image_path = os.path.join(media_images_file_path, url_decrypted + extention)
    if not os.path.exists(image_path):
        dic = {
            "caption": url,
            "url": url
        }
        failed_images.append(dic)

failed_images_num = len(failed_images)
print(f"failed_images_num:\t{failed_images_num}")

def caption_image(query, sample, phar):
    url = sample["url"]
    base_name, extention = os.path.splitext(url)
    url_decrypted = encode_url(url)
    file_path = os.path.join(media_images_caption_path, url_decrypted + ".json")
    if os.path.exists(file_path):
        phar.update(1)
        return
    image_path = os.path.join(media_images_file_path, url_decrypted + extention)
    if os.path.exists(image_path):
        base64_image = rc_image(image_path)
        caption = visual_requests(query, base64_image, api_model)
        item_dic = {
            "url": url,
            "caption": caption
        }
        save_dic2json(file_path, item_dic)
    phar.update(1)

# 多线程请求
total_num = len(media_images_config)
phar = tqdm(total=total_num)

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    for i in media_images_config:
        future = executor.submit(caption_image, query, i, phar)

