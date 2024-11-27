import json
import concurrent.futures
import threading
import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm.auto import tqdm
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from src.llm_backend import LLMBackend
from src.utils import *

max_workers = 30
MAX_RETRY_TIMES = 5

data_path = "data"
logs_path = "logs"

deduces_path = os.path.join(data_path, "deduces")
deduce_queries_path = os.path.join(data_path, "deduce_queries")

assert os.path.exists(deduce_queries_path)

check_dirs(deduces_path)

# 定义接收的数据格式
class DeduceInfo(BaseModel):
    rmr: bool = Field(description="Whether my Post shows the content of my profile.")
    ept: bool = Field(description="Whether my Post provides explicit evidence of my personality traits.")
    bpa: str = Field(description="A brief related psychological activities when I post my Post.")

# 创建输出解析器
output_parser = PydanticOutputParser(pydantic_object=DeduceInfo)

# 获取输出格式指示
format_instructions = output_parser.get_format_instructions()

# 打印提示
print("输出格式：", format_instructions)

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
# 写入字典到 JSON 文件
def save_dic2json(file_path, dic, check_exist=True):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dic, f, ensure_ascii=False, indent=4)

def deduce(query, phar):
    q_id = query["id"]
    q_content = query["content"]
    try_time = 0
    while try_time < MAX_RETRY_TIMES:
        try:
            output = llm.request(q_content)
            file_path = os.path.join(deduces_path, f"{q_id}.json")
            if os.path.exists(file_path):
                break
            # 解析模型的输出
            parsed_output = output_parser.parse(output)
            # 将 Pydantic 格式转换为字典
            parsed_output_dict = parsed_output.dict()
            # 打印字典
            # print("输出的数据：", parsed_output_dict)
            parsed_output_dict["id"] = q_id
            save_dic2json(file_path, parsed_output_dict)
            break
        except Exception as e:
            print(e)
            print(re)
            try_time += 1
    phar.update(1)

# 遍历 deduces_path 文件夹内的文件
already_duduced = []
for filename in tqdm(os.listdir(deduces_path)):
    # 检查文件是否为 JSON 文件
    if filename.endswith('.json'):
        file_path = os.path.join(deduces_path, filename)
        base_name, extension = os.path.splitext(filename)
        already_duduced.append(base_name)

print(f"already_duduced numbers:\t{len(already_duduced)}")

# 遍历 deduces_path 文件夹内的文件
deduces = []
for filename in tqdm(os.listdir(deduce_queries_path)):
    # 检查文件是否为JSON文件
    if filename.endswith('.json'):
        base_name, extension = os.path.splitext(filename)
        if base_name in already_duduced:
            continue
        file_path = os.path.join(deduce_queries_path, filename)
        with open(file_path, 'r', encoding="utf8") as f:
            sample = json.load(f)
            deduces.append(sample)

print(f"left duduce sample numbers:\t{len(deduces)}")

# 多线程请求
toal_deduce_num = len(deduces)
phar_deduce = tqdm(total=toal_deduce_num)
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    for i in deduces:
        future = executor.submit(deduce, i, phar_deduce)

