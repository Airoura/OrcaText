import os
import re
import json
import base64

from tqdm.auto import tqdm

def encode_url(url):  
    return base64.b64encode(url.encode('utf-8')).decode('utf-8')  

def decode_url(encoded_url):
    return base64.b64decode(encoded_url).decode('utf-8')

def check_dirs(directory):
   if not os.path.exists(directory):
       os.makedirs(directory)

def load_json(r):
    def get_elements_between_braces(s):
        start = s.find('{')
        end = s.rfind('}')
        if start == -1 or end == -1:
           return None
        return s[start:end + 1]
    result = get_elements_between_braces(r)
    return json.loads(result)
    
def read_users_info(user_info_path):
    # 遍历 user-info-map 文件夹内的文件
    users_info = {}
    for filename in tqdm(os.listdir(user_info_path)):
        # 检查文件是否为JSON文件
        if filename.endswith('.json'):
            file_path = os.path.join(user_info_path, filename)
            with open(file_path, 'r', encoding="utf8") as f:
                user_info = json.load(f)
            base_name, extension = os.path.splitext(filename)
            users_info[base_name] = user_info
    # print(f"len users:\t{len(users_info.keys())}")
    return users_info

def read_user_traits(user_summaries_path):
    # 遍历 user-traits 文件夹内的文件
    user_traits = {}
    for filename in tqdm(os.listdir(user_summaries_path)):
        # 检查文件是否为JSON文件
        if filename.endswith('.json'):
            file_path = os.path.join(user_summaries_path, filename)
            with open(file_path, 'r', encoding="utf8") as f:
                user_trait = json.load(f)
            base_name, extension = os.path.splitext(filename)
            user_traits[base_name] = user_trait
    
    # print(f"len users:\t{len(user_traits.keys())}")
    return user_traits

def read_user_posts(user_posts_path):
    user_posts = {}
    # 遍历文件夹内的文件
    for filename in tqdm(os.listdir(user_posts_path)):
        # 检查文件是否为 JSON 文件
        if filename.endswith('.json'):
            file_path = os.path.join(user_posts_path, filename)
            with open(file_path, 'r', encoding="utf8") as f:
                posts = json.load(f)
            if len(posts) >= 200:
                # 拆分文件名，取最后一个 "-" 后面的部分作为键
                # 使用 os.path.splitext 去掉拓展名
                base_name, extension = os.path.splitext(filename)
                user_posts[base_name] = posts
    return user_posts

def read_user_conversations(user_conversations_path):
    user_conversations = {}
    # 遍历文件夹内的文件
    for filename in tqdm(os.listdir(user_conversations_path)):
        # 检查文件是否为 JSON 文件
        if filename.endswith('.json'):
            file_path = os.path.join(user_conversations_path, filename)
            with open(file_path, 'r', encoding="utf8") as f:
                conversations = json.load(f)
            base_name, extension = os.path.splitext(filename)
            user_conversations[base_name] = conversations
    return user_conversations

def read_posts_id_map(posts_path):
    post_ids = []
    # 遍历 id-post-map 文件夹内的文件
    for filename in tqdm(os.listdir(posts_path)):
        # 检查文件是否为JSON文件
        if filename.endswith('.json'):
            base_name, extension = os.path.splitext(filename)
            post_ids.append(base_name)
    return post_ids

def read_user_profiles(user_profiles_path):
    user_profiles = {}
    # 遍历文件夹内的文件
    for filename in tqdm(os.listdir(user_profiles_path)):
        # 检查文件是否为JSON文件
        if filename.endswith('.json'):
            file_path = os.path.join(user_profiles_path, filename)
            with open(file_path, 'r', encoding="utf8") as f:
                profile = json.load(f)
                base_name, extension = os.path.splitext(filename)
                user_profiles[base_name] = profile
                
    return user_profiles

def read_post_knowledges(post_knowledges_path):
    post_knowledges = {}
    failed_count = 0
    # 遍历文件夹内的文件
    for filename in tqdm(os.listdir(post_knowledges_path)):
        # 检查文件是否为 JSON 文件
        if filename.endswith('.json'):
            file_path = os.path.join(post_knowledges_path, filename)
            with open(file_path, 'r', encoding="utf8") as f:
                try:
                    knowledges = json.load(f)
                except Exception as e:
                    print(file_path)
                    print(e)
                    failed_count += 1
                    if failed_count > 10:
                        raise "Too many failed samples."
            base_name, extension = os.path.splitext(filename)
            post_knowledges[base_name] = knowledges
                
    return post_knowledges


def read_real_selected_original_users(real_selected_original_users_path):
    if os.path.exists(real_selected_original_users_path):
        with open(real_selected_original_users_path, 'r', encoding="utf8") as f:
            real_selected_original_users = json.load(f)
        return real_selected_original_users
    else:
        return []
        
def read_user_avg_scores(user_avg_scores_path):
    with open(user_avg_scores_path, 'r', encoding="utf8") as f:
        user_avg_scores = json.load(f)
    return user_avg_scores


def read_user_explanations(user_explanations_path):
    with open(user_explanations_path, 'r', encoding="utf8") as f:
        user_explanations = json.load(f)
    return user_explanations


def read_user_traits(user_traits_path):
    user_traits = {}
    # 遍历文件夹内的文件
    for filename in tqdm(os.listdir(user_traits_path)):
        # 检查文件是否为JSON文件
        if filename.endswith('.json'):
            file_path = os.path.join(user_traits_path, filename)
            with open(file_path, 'r', encoding="utf8") as f:
                traits = json.load(f)
                base_name, extension = os.path.splitext(filename)
                user_traits[base_name] = traits
    return user_traits


def save_dic2json(file_path, dic, check_exist=True):
    # 写入结果到文件
    with open(file_path, 'w', encoding='utf-8') as f:
        # 使用 OrderedDict 或排序后的 dict
        json.dump(dic, f, ensure_ascii=False, indent=4)

def append_captions(media, media_images_caption_path):
    captions = []
    for i, image_url in enumerate(media):
        if "https://pbs.twimg.com" not in image_url:
            print(image_url)
        assert("https://pbs.twimg.com" in image_url)
        base_name, extention = os.path.splitext(image_url)
        url_decrypted = encode_url(image_url)
        caption_path = os.path.join(media_images_caption_path, url_decrypted + ".json")
        if os.path.exists(caption_path):
            try:
                with open(caption_path, 'r', encoding="utf8") as f:
                    # print(caption_path)
                    meta_info = json.load(f)
                    caption = meta_info["caption"]
            except Exception as e:
                print(caption_path)
                print(e)
                return []
            dic = {
                "type": "image",
                "content": caption
            }
            captions.append(dic)
    return captions

def convert_quote_media_to_array(post):
    result = []
    try:
        if "entities" in post:
            entities = post["entities"]
        elif "extendedEntities" in quote:
            entities = post["extendedEntities"]
        else:
            return media
        if "media" in entities:
            media = entities["media"]
            for m in media:
                result.append(m["media_url_https"])
    except Exception as e:
        print(e)
    return result

def process_conversation(user_conversations_path, user_posts, post_ids, posts_path, media_images_caption_path):
    user_conversations = {}
    for user, posts in tqdm(user_posts.items()):
        conversations = []
        for ind, post in enumerate(posts):
            if post:
                name = post["author"]["name"]
                text = post["text"]
                p_id = post["id"]
                isReply = post["isReply"]
                isQuote = post["isQuote"]
                url = post["url"]
                media_r = post["media"]
                captions_r = append_captions(media_r, media_images_caption_path)
                
                post_quote = {}
                if "quote" in post:
                    temp_quote = post["quote"]
                    # 处理 quote 中的媒体资源
                    media_rq = convert_quote_media_to_array(temp_quote)
                    captions_rq = append_captions(media_rq, media_images_caption_path)
                    post_quote = {
                        "Quote User": temp_quote["author"]["name"],
                        "Quote Text": temp_quote["text"],
                        "Quote Media": captions_rq
                    }
                if len(text) < 50 and not post_quote:
                    continue
                conversation = [
                    {
                        "User": name, 
                        "Post": text,
                        "Quote": post_quote,
                        "Media": captions_r
                    }
                ]
                dic = {
                    "meta_data": {
                        "user": user,
                        "id": p_id,
                        "url": url,
                        "conve_id": f"{user}-{ind}",
                        "isQuote": isQuote,
                        "isReply": isReply,
                        "reply_meta_data": {}
                        
                    },
                    "conversation": conversation
                }
                if isReply:
                    inReplyToUsername = post["inReplyToUsername"]
                    try:
                        if "inReplyToId" in post:
                            inReplyToId = post["inReplyToId"]
                        else:
                            inReplyToId = post["inReplyToUserId"]
                    except Exception as e:
                        print(e)
                        print(upstream_post)
                        continue
                    if inReplyToId in post_ids:
                        upstream_post_path = os.path.join(posts_path, f"{inReplyToId}.json")
                        with open(upstream_post_path, 'r', encoding="utf8") as f:
                            upstream_post = json.load(f)
                        
                        # 如果是一个thread，抛弃这条数据，因为上下文信息不完整
                        try:
                            upStreamIsReply = upstream_post["isReply"]
                            if upStreamIsReply:
                                underThreadId = None
                                if "inReplyToId" in upstream_post:
                                    underThreadId = upstream_post["inReplyToId"]
                                if "inReplyToUserId" in upstream_post:
                                    underThreadId = upstream_post["inReplyToUserId"]
                                if underThreadId:
                                    continue
                            # 获取上游推特内容
                            upstream_user = upstream_post["author"]["name"]
                            if upstream_user == name:
                                continue
                            upstream_url = upstream_post["url"]
                            upstream_id = upstream_post["id"]
                            upstreamIsQuote = upstream_post["isQuote"]
                            upstream_meta_data = {
                                "user": upstream_user,
                                "id": p_id,
                                "url": upstream_url,
                                "isReply": upStreamIsReply,
                                "isQuote": upstreamIsQuote
                            }
                            dic["meta_data"]["reply_meta_data"] = upstream_meta_data
                            upstream_text = upstream_post["text"]
                            media_u = upstream_post["media"]
                            captions_u = append_captions(media_u, media_images_caption_path)
                            
                            upstream_quote = {}
                
                            if "quote" in upstream_post:
                                temp_quote = upstream_post["quote"]
                                # 处理 quote 中的媒体资源
                                media_uq = convert_quote_media_to_array(temp_quote)
                                captions_uq = append_captions(media_uq, media_images_caption_path)
                                upstream_quote = {
                                    "Quote User": temp_quote["author"]["name"],
                                    "Quote Text": temp_quote["text"],
                                    "Quote Media": captions_uq
                                }
                                
                            if len(text) < 50 and not upstream_quote:
                                continue
                                
                            upstream_dic = {
                                "User": upstream_user, 
                                "Content": upstream_text,
                                "Quote": upstream_quote,
                                "Media": captions_u
                            }
                            dic["conversation"].insert(0, upstream_dic)
                        except Exception as e:
                            print(e)
                            print(upstream_post)
                            continue
                conversations.append(dic)
        user_conversations[user] = conversations

    check_dirs("data/preview")
    save_dic2json("data/preview/user_conversations.json", user_conversations)
    check_dirs(user_conversations_path)
    for user, conversation in tqdm(user_conversations.items()):
        file_path = os.path.join(user_conversations_path, f"{user}.json")
        save_dic2json(file_path, conversation)

def read_user_scores(user_scores_path):
    # 指定目录路径
    directory_path = user_scores_path
    # 获取目录中的文件列表
    files = os.listdir(directory_path)
    # 过滤出只包含文件的列表（不包含目录）
    files = [f for f in files if os.path.isfile(os.path.join(directory_path, f))]  
    # 对文件名进行排序
    sorted_files = sorted(files)  
    user_scores = {}
    # 遍历文件夹内的文件
    for filename in tqdm(sorted_files):
        # 检查文件是否为JSON文件
        if filename.endswith('.json'):
            file_path = os.path.join(user_scores_path, filename)
            with open(file_path, 'r', encoding="utf8") as f:
                score = json.load(f)
            # 使用os.path.splitext去掉拓展名
            base_name, extension = os.path.splitext(filename)
            user_scores[base_name] = score
    return user_scores

def replace_alt(text):
    # text = '@nicochristie @a16zGames @speedrun @Altera_AL Wooo!!! Congrats'  
    # 首先找到第一个@及其后面的内容，并用@[userName]替换  
    # 然后删除其他所有的@及其后面的内容  
    first_at_pattern = r'(@[\w_]+)'  
    rest_at_pattern = r'@[\w_]+'  
      
    # 先用@[userName]替换第一个@及其后面的内容  
    first_replacement = '@[userName]'  
    first_replacement = ''
    first_replaced = re.sub(first_at_pattern, first_replacement, text, count=1)  
      
    # 然后删除其他所有的@及其后面的内容  
    rest_replacement = ''  
    final_result = re.sub(rest_at_pattern, rest_replacement, first_replaced)  
    return final_result
    
def replace_url(text):
    # 假设这是你的原始字符串  
    # text = "Promising start for Squad Busters! 🔥🔥🔥🔥 https://t.co/ppVcMsBpsA"  
    
    # 定义一个正则表达式模式来匹配常见的post URL  
    # 注意：这个模式可能不是完美的，但它应该能够匹配大多数常见的post URL  
    post_url_pattern = r'https?://t\.co/[a-zA-Z0-9]+'  
      
    # 使用re.sub方法来替换匹配的URL  [post_url]
    replaced_text = re.sub(post_url_pattern, '', text)  
    
    return replaced_text

def post_cleaner(content):
    filterd_content = replace_url(replace_alt(content)).strip()
    filterd_content = re.sub(r'\s+', ' ', filterd_content)
    return filterd_content






