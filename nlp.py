# python -m spacy download en_core_web_sm
# pip install transformers[sklearn] --force-reinstall

from typing import List, Dict, Union, Tuple
from transformers import pipeline
import spacy
import re
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

class Document():
    def __init__(self, full_text: str, id: str, metadata: str = None):
        '''
        `doc` is the document's text
        `metadata` is any parsable string holding some metadata
        `id` is meant to have a meaning in the source system
        '''
        self.document = full_text
        self.id = id
        self.metadata = metadata
        self.passages = []
        self.embeddings = []
        self.bow = []
    
    def size(self):
        return len(self.passages)

class Question():
    def __init__(self, question, embedding, bow):
        self.question = question
        self.embedding = embedding
        self.bow = bow
    
class PassageTokenizer():
    def __init__(self, window_size: int = 3, step: int = 2):
        '''
        Tokenize passages of `windows_size` sentences
        overlapped by `step` sentences
        '''
        self.nlp = spacy.load("en_core_web_sm")
        self.window_size = window_size
        self.step = step
    
    def get_passages(self, document: str) -> List[str]:
        '''
        Extract all passages from `document`
        '''
        doc_passages = []
        doc = self.nlp(document)
        sentences = [str(s) for s in doc.sents]
        for i in range(0, len(sentences), self.step):
            _t = " ".join(sentences[i:i+self.window_size])
            doc_passages.append(_t)
        
        return doc_passages

class BagWords():
    def __init__(self, corpus: List[str]=None, stop_words: List[str]="english", ngram_range: Tuple = (1,1)):
        '''
        Initialize a Tf-Idf vectorizer and fit it with `corpus` is provided
        '''
        if corpus:
            self.vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range = ngram_range)
            self.vectorizer.fit_transform(corpus)
        else:
            self.vectorizer = None
    
    def save(self, root: str):
        '''
        Store a class object state, to be loaded with the `load` class method
        '''
        path = os.path.join(root, 'bow.pkl')
        with open(path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
    
    @classmethod
    def load(cls, path):
        '''
        Instantiate a class object from a previously stored object state, using `save`
        '''
        with open(path, 'rb') as f:
            vectorizer = pickle.load(f)
            bow = cls()
            bow.vectorizer = vectorizer
        return bow    
    
    def transform(self, text: Union[List[str], str]):
        '''
        Transform `text` and return its tf-idf representation
        '''
        if type(text) == str:
            text = [text]
        return self.vectorizer.transform(text)
        
    def similarity(self, x, y) -> List[float]:
        '''
        Return cosine similarities between `question` and `passages`
        '''
        cosine_similarities = linear_kernel(x, y).flatten()
        return cosine_similarities[0]
    
    def __call__(self, text: Union[List[str], str]):
        return self.transform(text)

class QA():
    def __init__(self, model: str = None):
        '''
        `model` is the name of an HaggingFace model for reading comprehension, or a 
        local path to a fine-tuned model, that can be loaded with `from_pretrained`.
        Implementation is using https://huggingface.co/tasks/question-answering
        '''
        if model:
            self.model = pipeline(model=model)
        else:
            self.model = pipeline("question-answering")
                
    
    def get_answer_from_passage(self, question: str, passage: str) -> List[Dict[str, float]]:
        ans = self.model(question=question, context=passage.question)
        ## example of answer would be {'answer': 'İstanbul', 'end': 39, 'score': 0.953, 'start': 31}
        return ans['score']
        
    def __call__(self, question: str, passages: List[str], min_relevance = None) -> List[Dict[str, float]]:
        return self.get_answer_from_passage(question, passages)


