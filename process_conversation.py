import os

from src.utils import read_user_posts, read_posts_id_map, process_conversation

data_path = "data"

posts_path = os.path.join(data_path, "id_post_map")
user_posts_path = os.path.join(data_path, "user_posts_map")

media_images_caption_path = os.path.join(data_path, "captions")
user_conversations_path = os.path.join(data_path, "user_conversations")

user_posts = read_user_posts(user_posts_path)
post_ids = read_posts_id_map(posts_path)

process_conversation(user_conversations_path, user_posts, post_ids, posts_path, media_images_caption_path)