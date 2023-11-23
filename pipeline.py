from typing import List, Dict
from transformers import pipeline
import spacy
import re
import pandas as pd
import pickle
import os
from nlp import Document, Question, PassageTokenizer, BagWords, QA
from llm import Embedder, Agent
from openai.embeddings_utils import cosine_similarity
from tqdm import tqdm 

class Cache():
    '''
    This is just a dictionary of documents. Should be
    a service, such as Az Table or Cosmos
    '''
    def __init__(self, documents: Dict = None):
        documents = documents if documents else {}
        self.documents = documents
    
    def set_document(self, doc: Document):
        self.documents[doc.id] = doc

    def get_document(self, id: str):
        if id in self.documents.keys():
            return self.documents[id]
        else:
            return None
    
    def save(self, root: str):
        path = os.path.join(root, 'cache.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self.documents, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            documents = pickle.load(f)
        return cls(documents)


class ConversationPipeline():
    
    def __init__(self, 
                    cache_path: str = None, 
                    bow_path: str = None,
                    bow_corpus: List[str] = None,
                    similarity_threshold: float = 0.5, 
                    qa_threshold: float = 0.5, 
                    do_sem_similairty: bool = True, 
                    do_sin_similairty: bool = True, 
                    do_qa: bool = True):
    
        
        if bow_path:
            self.bow = BagWords.load(bow_path)  
        elif bow_corpus:
            self.bow = BagWords(bow_corpus)
        else:
            raise("ERROR: need to init BOW somehow")
        self.cache = Cache.load(cache_path) if cache_path else Cache()
        self.tok = PassageTokenizer() # missing arguments
        self.agent = Agent() # missing arguments
        self.embedder = Embedder() # missing arguments
        self.qa = QA() # missing arguments
        self.similarity_threshold = similarity_threshold
        self.qa_threshold = qa_threshold
        self.do_sem_similairty = do_sem_similairty
        self.do_sin_similairty = do_sin_similairty
        self.do_qa = do_qa

    def encode_document(self, full_text: str, id: str, metadata: str):
        doc = self.cache.get_document(id)
        if doc is None:
            doc = Document(full_text, id, metadata)
            doc.passages = self.tok.get_passages(doc.document)
            doc.embeddings = self.embedder(doc.passages)
            doc.bow = self.bow.transform(doc.passages)
            self.cache.set_document(doc)
        return doc
    
    def encode_question(self, question: str):
        emb = self.embedder(question)
        bow = self.bow.transform(question)
        return Question(question, emb, bow)


    def get_best_passages(self, doc: Document, question: Question) -> List[Dict]:
        passages = []

        # check semantic similarity
        if self.do_sem_similairty:
            sem_sims = [self.embedder.similarity(question.embedding, d) for d in doc.embeddings]
        else:
            sem_sims = [0]*doc.size

        # check sintax similarity
        if self.do_sin_similairty:
            sin_sims = [self.bow.similarity(question.bow, d) for d in doc.bow]
        else:
            sin_sims = [0]*doc.size
        
        sims = [max(sem, sim) for sem, sim in zip(sem_sims, sin_sims)]
        for p, s in zip(doc.passages, sims):
            if s < self.similarity_threshold:
                continue
            
            passage = {'doc_id': doc.id, 'metadata': doc.metadata, 'passage': p} #doc.passages[idx]
            # check reading comprehension
            if self.do_qa:
                score = self.qa(p, question) #doc.passages[idx]
                if score < self.qa_threshold:
                    continue
                passage['score'] = score
            else:
                passage['score'] = s
            
            passages.append(passage)
        
        return passages

    def answer_question(self, documents: List[Dict], question: str, as_dict: bool = False) -> str: 
        passages = []
        q = self.encode_question(question)
        for d in tqdm(documents):
            doc = self.encode_document(d['doc'], d['id'], d['metadata'])
            doc_passages = self.get_best_passages(doc, q)
            passages.extend(doc_passages)
        
        prompt, answer = self.agent(question, [p['passage'] for p in passages])
        

        if as_dict:
            answer_dict = {'answer': answer}
            answer_dict['prompt'] = prompt
            answer_dict['passages'] = passages
            return answer_dict
        
        return answer


if __name__ == '__main__':
    from sklearn.datasets import fetch_20newsgroups

    # params
    path = './documents.csv'
    question = 'What size were the doors of the Bricklin car?'
    similarity_threshold = 0.3
    qa_threshold = 0.3
    
    # we are assuming a search service returns a list of documents,
    # each having a system ID, used to match the cache, and metadata 
    # to be return to the user as a consumable reference to the document
    documents = pd.read_csv(path)
    assert set(documents.columns) & {'doc', 'id'} == {'doc', 'id'}
    documents['metadata'] = documents['id'].apply(lambda r: f"bla-{r}")
    agent = ConversationPipeline(bow_corpus=fetch_20newsgroups().data)
    answer = agent.answer_question(documents.to_dict('records'), question, as_dict = True)
    print(answer)