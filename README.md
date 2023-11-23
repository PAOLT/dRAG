# dRAG

To install the environment:

```
conda env create -f conda.yaml
conda activate drag
python -m spacy download en_core_web_sm
```

It might be necessary to run 
`pip install transformers[sklearn] --force-reinstall`

For quick examples of usage see `examples.ipynb`