from typing import List, Any, Dict

class Search():
    def __init__(self, documents: Any, num_documents: int = 10):
        self.documents = documents
        self.num_documents = num_documents

    def search_ids(self, question: str = None) -> List[str]:
        retrieved_documents = self.documents.sample(self.num_documents).to_dict('records')
        doc_ids = [{'id': doc['id'], 'date': doc['date']} for doc in retrieved_documents]
        return doc_ids
    
    def retrieve_docs(self, doc_ids: List[str], in_cache: List[bool]) -> Dict:
        '''
        Retrieve documents from PortalOne, with `doc_ids`
        We simulate retrieving one document at a time from PortalOne
        The best retrieve strategy should be identified.
        '''
        ids_to_retrieve = [d for d, c in zip(doc_ids, in_cache) if not c]        
        for id in ids_to_retrieve:
            yield self.documents[self.documents.id==id].to_dict('records')[0]
