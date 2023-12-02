from typing import Dict, List, Union
from azure.data.tables import TableServiceClient
from abc import ABC, abstractmethod
import numpy as np

class AbstractCache(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def set_document(self, doc_id: str, metadata: str, doc_passages: List[Dict]) -> Union[str,None]:
        '''
        Creates a set of document passage entities, partitioned over `doc_id`.
        Return `doc_id` if the document is succesfully created
        '''
        pass

    @abstractmethod
    def get_document(self, id: str) -> Union[Dict, None]:
        '''
        Return the `doc_id` if the document exists, `None` otherwiose.
        '''
        pass

    
    @abstractmethod
    def get_passages(self, id: str) -> List[Dict]:
        '''
        Return passages for document `id`
        '''
        pass
        
    
    @abstractmethod
    def check_documents(self, docs_ids: List[Dict[str,str]]) -> List[bool]:
        '''
        Returns if a list of documents is in the cache
        '''
        pass
        



class SimpleCache(AbstractCache):
    '''
    This is just a dictionary of documents. Should be
    a service, such as Az Table or Cosmos
    '''
    def __init__(self):
        self.documents = {}
        self.passages = {}
    
    def set_document(self, doc_id: str, metadata: str, doc_passages: List[Dict]):
        try:
            guids = [p['id'] for p in doc_passages]
            self.documents[doc_id] = guids
            for passage in doc_passages:
                self.passages[passage['id']] = {'passage':passage['passage'], 'emb':passage['emb'], 'metadata': metadata, 'doc_id': doc_id}
            return doc_id
        except:
            return None

    def get_document(self, id: str) -> Dict:
        try:
            doc = self.documents.get(id, None)
            return doc['id']
        
        except:
            return None
    
    def get_passages(self, id: str):
        try:
            guids = self.documents[id]
            passages = [self.passages[guid] for guid in guids]
            return passages
        
        except:
            return None 
        
    def check_documents(self, docs_ids: List[Dict[str,str]]) -> List[bool]:
        checks = []
        for doc_id in docs_ids:
            cached_doc = self.get_document(doc_id['id'])
            in_cache = doc_id['date'] <= cached_doc['date'] if cached_doc else False
            checks.append(in_cache)
        return checks
    

class Cache(AbstractCache):
    '''
    This is just a dictionary of documents. Should be
    a service, such as Az Table or Cosmos
    '''
    def __init__(self, conf: Dict = None):
        connection_string = f"AccountName={conf['account_name']};AccountKey={conf['account_key']};EndpointSuffix=core.windows.net"
        try:
            table_service_client = TableServiceClient.from_connection_string(conn_str=connection_string)
            self.documents_client = table_service_client.get_table_client(table_name='p1documents')
            self.passages_client = table_service_client.get_table_client(table_name='p1passages')
        except:
            raise("Error retrieving tables")

    
    def set_document(self, doc: Dict, n_passages: List):
        try:
            doc['RowKey'] = doc.pop('id')
            doc['PartitionKey'] = '0'
            doc['n_passages'] = n_passages
            _ = doc.pop('doc', None)
            entity = self.documents_client.create_entity(entity=doc)
            return entity
        
        except Exception as e:
            raise(f"Failed to create entity:\n{e}")

    def get_document(self, id: str) -> Dict:
        try:
            doc_query = f"RowKey eq '{id}'"
            doc = self.documents_client.query_entities(doc_query)
            return next(doc, None) 
        
        except Exception as e:
            raise(f"Failed to retrieve document {id}:\n{e}")
    
    def set_passage(self, doc_id: str, passage_id: str, passage: Dict):
        '''
        Store a passage in the cache using `doc_id` as PartitionKey and `passage_id` as RowKey
        '''
        try:
            passage['PartitionKey'] = str(doc_id)
            passage['RowKey'] = str(passage_id)
            emb = np.array(passage.pop('emb')).tostring()
            passage['emb'] = emb
            entity = self.passages_client.create_entity(entity=passage)
            return entity
        except Exception as e:
            raise(f"Failed to create passage entity {passage_id} for document {doc_id}:\n{e}")

    def get_passages(self, id: str):
        try:
            passages_query = f"PartitionKey eq '{id}'"
            passages = self.passages_client.query_entities(passages_query)
            for passage in passages:
                emb = np.fromstring(passage.pop('emb'))
                passage['emb']=emb
                yield passage

        except Exception as e:
            raise(f"Failed to retrieve passages for document {id}:\n{e}")

    def get_doc_corpus(self, id: str):
        try:
            corpus = []
            passages_query = f"PartitionKey eq '{id}'"
            passages = self.passages_client.query_entities(passages_query)
            for passage in passages:
                corpus.append(passage['passage'])
            return corpus
        
        except Exception as e:
            raise(f"Failed to retrieve passages for document {id}:\n{e}")
        
    def check_documents(self, docs_ids: List[Dict[str,str]]) -> List[bool]:
        checks = []
        for doc_id in docs_ids:
            cached_doc = self.get_document(doc_id['id'])
            in_cache = doc_id['date'] <= cached_doc['date'] if cached_doc else False
            checks.append(in_cache)
        return checks
