# python -m spacy download en_core_web_sm
# pip install transformers[sklearn] --force-reinstall


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
    
