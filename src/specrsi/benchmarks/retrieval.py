"""Frozen full Wikipedia retrieval for the GEPA HoVer three-hop objective."""
from pathlib import Path
import json
from .sandbox import ROOT
from .base import MissingArtifactError

class WikipediaRetriever:
    def __init__(self,directory:Path|None=None):
        import bm25s,Stemmer
        directory=directory or ROOT/'data/retrieval/hover'
        if not (ROOT/'reports/hover_retrieval.json').is_file(): raise MissingArtifactError('Run scripts/prepare_retrieval.py first')
        self.index=bm25s.BM25.load(str(directory/'bm25s'),mmap=True)
        self.stemmer=Stemmer.Stemmer('english')
        self.passages=[json.loads(line) for line in (directory/'passages.jsonl').open()]
    def search(self,query:str,k:int)->list[str]:
        import bm25s
        tokens=bm25s.tokenize(query,stopwords='en',stemmer=self.stemmer,show_progress=False)
        rows,_=self.index.retrieve(tokens,k=k,n_threads=1,show_progress=False)
        return [self.passages[int(i)] for i in rows[0]]
