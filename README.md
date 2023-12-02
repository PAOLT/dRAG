# dRAG
**This package is not meant for production use, only for evaluation**

To install the environment:

```
conda env create -f conda.yaml
conda activate drag
python -m spacy download en_core_web_sm
```

It might be necessary to run 
`pip install transformers[sklearn] --force-reinstall`

To run the code, do the following:

- Create a Storage Account in Azure
- Create two tables named `p1documents` and `p1passages`
- Create the two files `aoai_conf.yml` and `table_conf.yml` with the required information to access to OpenAI and Azure Storage
- To connect the library to PortalOne, an implementation of the `Search` class must be provided.

Run the file `test.py` (note it holds some parameters at the beginning)

