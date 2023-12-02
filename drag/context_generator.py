from typing import List, Dict, Tuple
from .bow import BagWords
from .llm import Embedder
from .cache import Cache


class ContextGenerator():
    
    def __init__(self, 
                doc_ids: List, # full documents from the search
                cache: Cache, 
                embedder: Embedder,
                similarity_threshold: float = 0.5):      

        corpus = []
        for doc_id in doc_ids:
            doc_corpus = cache.get_doc_corpus(doc_id)
            corpus.extend(doc_corpus)
        self.bow = BagWords(corpus)
        self.cache = cache
        self.similarity_threshold = similarity_threshold   
        self.embedder = embedder
        self.doc_ids = doc_ids
    
    def encode_question(self, question: str):
        emb = self.embedder(question)
        bow = self.bow.transform(question)
        return (emb, bow)

    def best_doc_passages(self, doc_id: Dict, q: Tuple) -> List[Dict]:
        passages, sem_sims, sin_sims, doc_passages = [], [], [], []

        for passage in self.cache.get_passages(doc_id): #doc_passages:
            sem_sims.append(self.embedder.similarity(q[0], passage['emb']))
            sin_sims.append(self.bow.similarity(q[1], self.bow.transform(passage['passage'])))
            doc_passages.append(passage['passage'])
            
        sims = [max(sem, sim) for sem, sim in zip(sem_sims, sin_sims)]
        for p, s in zip(doc_passages, sims):
            if s < self.similarity_threshold:
                continue            
            passage = {'passage': p, 'score': s} 
            passages.append(passage)
        
        return passages

    def passages_from_question(self, question: str) -> str: 
        all_best_passages = []
        q = self.encode_question(question)
        for doc_id in self.doc_ids:
            best_passages = self.best_doc_passages(doc_id, q)
            if best_passages:
                all_best_passages.extend(best_passages)
        
        return all_best_passages
    
    def __call__(self, question: str) -> str: 
        return self.passages_from_question(question) 
