from typing import List, Union, Tuple
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


class BagWords():
    def __init__(self, corpus: List[str], stop_words: List[str]="english", ngram_range: Tuple = (1,1)):
        '''
        Initialize a Tf-Idf vectorizer and fit it with `corpus` is provided
        '''
        self.vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range = ngram_range)
        self.vectorizer.fit_transform(corpus)
    
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
    
    def __call__(self, x,y):
        return self.similarity(x,y)

