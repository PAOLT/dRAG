from sklearn.datasets import fetch_20newsgroups
from drag import Cache, Embedder, PassageTokenizer, ContextGenerator, Search
import pandas as pd
import yaml
from time import time
from datetime import datetime


# params
path = './documents.csv'
num_documents = 10
question = 'What size were the doors of the Bricklin car?'
similarity_threshold = 0.5

try:
    documents = pd.read_csv(path)
except: 
    docs = []
    date = datetime.now().strftime('%Y-%m-%d') # bring it back with datetime.strptime(date, '%Y-%m-%d').date()
    for n, doc in enumerate(fetch_20newsgroups().data):
        docs.append({'id':f"d_{n}", 'doc': doc, 'metadata': f"doc_{n}", 'date':date})
    documents = pd.DataFrame(docs)
    documents.to_csv(path, index=False)

########################

search = Search(documents=documents, num_documents=5)

with open('./table_config.yml', "r") as file:
        table_config = yaml.safe_load(file)
cache = Cache(table_config)

with open('./aoai_config.yml', "r") as file:
        aoai_config = yaml.safe_load(file)
embedder = Embedder(aoai_config) 

tokenizer = PassageTokenizer(window_size=5, step=2)

for _ in range(10):

    # retrieve documents
    doc_ids_dates = search.search_ids(question=question)
    doc_ids = [d['id'] for d in doc_ids_dates]
    print(f"\nRetrieved document IDs: {doc_ids}")

    # check the cache
    in_cache = cache.check_documents(doc_ids_dates)
    print(f"In cache: {in_cache}")

    if not any(in_cache):
         print("Caching some new documents")
    t0 = time()
    for doc in search.retrieve_docs(doc_ids, in_cache):
        n=0
        for passage in tokenizer.get_passages(doc['doc']):
            embeddings = embedder(passage)
            passage_id = n
            _passage = {'passage': passage, 'emb': embeddings}
            entity = cache.set_passage(doc_id = doc['id'], passage_id = passage_id, passage = _passage)
            n+=1
        entity = cache.set_document(doc=doc, n_passages=n)
    t1 = time()
    print(f"Elapsed for caching- {len([c for c in in_cache if not c])}/{len(in_cache)} documents: {t1-t0}")
    
    agent = ContextGenerator(doc_ids, cache, embedder, similarity_threshold)
    t0 = time()
    passages = agent(question)
    t1 = time()
    print(f"Elapsed for matching: {t1-t0}")
    print(f"Retrieved {len(passages)} passages")
    