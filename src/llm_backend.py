from zhipuai import ZhipuAI
from .glm4_api import ChatMessage, call_openai
from openai import OpenAI

class GLM4Private:
    def __init__(self, api_key="120985c00e389dac93ae62522ab5ae7a.lX6mrF4YEcSw4fmq", model="glm4"):
        self.model = model
        self.api_key=api_key
        
    def request(self, query, temperature=0.01, top_p=0.7):
        messages = []
        messages.append(ChatMessage(role = 'user', content = query))
        chat_openai = call_openai(self.model, 'test', debug = False)
        response_from_openai = chat_openai(
            messages = messages,
            temperature = temperature,
            top_p=top_p
        )
        return response_from_openai[0]
        
    def test(self, query="Hello LLMs!"):
        return self.request(query)
    
class LLMBackend:
    def __init__(self, platform, base_url, api_key, model, max_tokens=1024, temperature=0.6, top_p=0.7):
        self.api_key=api_key
        self.model=model
        self.max_tokens=max_tokens
        self.temperature=temperature
        self.top_p=top_p
        
        if platform=="zhipuai":
            self.client = ZhipuAI(
                base_url = base_url,
                api_key=self.api_key
            )
        else:
            self.client = OpenAI(
                base_url = base_url,
                api_key=self.api_key
            )
    
    def request(self, query):
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            messages = [
                {
                    "role": "user", 
                    "content": query
                }
            ]
        )
        content = response.choices[0].message.content
        return content

    def test(self, query="Hello LLMs!"):
        return self.request(query)

