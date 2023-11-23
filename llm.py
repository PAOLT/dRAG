from typing import List, Dict, Union
import re
import openai
import tiktoken
import yaml
from openai.embeddings_utils import get_embedding, cosine_similarity

       
default_system_message = '''You are a professional tax expert that answer user questions. 
A list of sentences is provided below, in triple squared brackets. They represent
independent facts. Use them to answer to user questions, do not use any external information.
The history of the conversation is also provided in triple brackets. You can use it for better
understanding the user question, but try to not be repetitive. If you are not sure about what 
to answer, or the facts do not comprise information for answering, simply answer 'I do not 
have enough information for answering, sorry'. Your style is formal and professional.
'''

class Agent():
    def __init__(self, openai_config_path = './aoai_conf.yml', system_message: str = default_system_message):
        
        self.system_message = system_message #if system_message else default_system_message
        self.previous_q = []
        self.previous_a = []

        with open(openai_config_path, "r") as file:
            aoai_config_data = yaml.safe_load(file)

        self.chat_engine = aoai_config_data['chat_model']
        
        openai.api_type = "azure"
        openai.api_key = aoai_config_data['api_key']
        openai.api_base = aoai_config_data['endpoint']
        openai.api_version = aoai_config_data['api_version']     
    
    def chat(self, question: str, context: List[str], as_dict=True):
        messages = []
        prompt_system = self.system_message + '\n\nFacts:\n[[['
        prompt_system += "\n-".join(context)
        prompt_system += "\n]]]"
        messages.append({"role": "system", "content": prompt_system})

        for q,a in zip(self.previous_q, self.previous_a):
           messages.append({"role": "user", "content": q})
           messages.append({"role": "assistant", "content": a})
        
        messages.append({"role": "user", "content": question})
        
        responses = openai.ChatCompletion.create(engine=self.chat_engine, messages=messages)
        answer = responses['choices'][0]['message']['content']
        self.previous_q.append(question)
        self.previous_a.append(answer)
        
        prompt = '\n'.join(messages)
        return prompt, answer
    
    def __call__(self, question: str, context: List[str]):
        return self.chat(question, context)

class Embedder():

    def __init__(self, openai_config_path = './aoai_conf.yml', max_input_size: int = 8192):
        
        # self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_input_size = max_input_size

        with open(openai_config_path, "r") as file:
            aoai_config_data = yaml.safe_load(file)

        self.emb_engine = aoai_config_data['emb_model']

        openai.api_type = "azure"
        openai.api_base = aoai_config_data['endpoint']
        openai.api_key = aoai_config_data['api_key']
        openai.api_version = aoai_config_data['api_version']

    
    def _preprocess(self, doc: str) -> str:        
        doc = doc.replace('\n', '')
        doc = re.sub(r'\s+', ' ', doc)
        doc = doc.strip()
        doc = doc[:self.max_input_size]
        return doc


    def embed(self, docs: Union[str, List[str]]):
        if type(docs) == str:
            doc = self._preprocess(docs)
            emb = get_embedding(doc, self.emb_engine)
            return emb
        else:
            docs = [self._preprocess(d) for d in docs]
            embs = [get_embedding(d, self.emb_engine) for d in docs]
            return embs
    
    def similarity(self, x: List[float] , y: List[float]) -> float:
        cs = cosine_similarity(x, y)
        return cs
    
    def __call__(self, docs: Union[str, List[str]]):
        return self.embed(docs)