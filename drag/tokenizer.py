# python -m spacy download en_core_web_sm
# pip install transformers[sklearn] --force-reinstall

from typing import List
import spacy

    
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
        doc = self.nlp(document)
        sentences = [str(s) for s in doc.sents]
        for i in range(0, len(sentences), self.step):
            _t = " ".join(sentences[i:i+self.window_size])
            yield _t
        




