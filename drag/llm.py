from typing import List, Union
import re
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity


class Embedder():

    def __init__(self, aoai_config, max_input_size: int = 8192):
        
        self.max_input_size = max_input_size
        self.emb_engine = aoai_config['emb_model']

        openai.api_type = "azure"
        openai.api_base = aoai_config['endpoint']
        openai.api_key = aoai_config['api_key']
        openai.api_version = aoai_config['api_version']

    
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