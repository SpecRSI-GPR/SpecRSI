#!/usr/bin/env python3
"""Prepare GEPA's pinned Wikipedia-2017 corpus and BM25 parameters locally."""
from pathlib import Path
import sys, json, tarfile, hashlib, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prepare_data import download
from specrsi.benchmarks.data import sha256_file

def main():
    import bm25s, Stemmer
    directory=ROOT/'data/retrieval/hover'; directory.mkdir(parents=True,exist_ok=True)
    revision='ef6a5e72a98b47cef31574a400fea8fe149559a3'
    url=f'https://huggingface.co/dspy/cache/resolve/{revision}/wiki.abstracts.2017.tar.gz'
    archive=directory/'wiki.abstracts.2017.tar.gz'
    download({'url':url},archive)
    corpus_path=directory/'wiki.abstracts.2017.jsonl'
    if not corpus_path.exists():
        with tarfile.open(archive) as tf:
            for member in tf:
                if Path(member.name).name=='wiki.abstracts.2017.jsonl' and member.isfile():
                    import shutil
                    with tf.extractfile(member) as source, corpus_path.open('wb') as out: shutil.copyfileobj(source,out)
                    break
    if not corpus_path.exists(): raise RuntimeError('Expected Wikipedia corpus not found')
    index=directory/'bm25s'; docs=directory/'passages.jsonl'
    if not (index/'params.index.json').is_file():
        corpus=[]
        with corpus_path.open() as source, docs.open('w') as out:
            for line in source:
                row=json.loads(line); passage=f"{row['title']} | {' '.join(row['text'])}"
                corpus.append(passage); out.write(json.dumps(passage,ensure_ascii=False)+'\n')
        print(f'Indexing {len(corpus)} documents',flush=True)
        tokens=bm25s.tokenize(corpus,stopwords='en',stemmer=Stemmer.Stemmer('english'))
        retriever=bm25s.BM25(k1=.9,b=.4); retriever.index(tokens); retriever.save(str(index))
    files=[archive,corpus_path,docs,*sorted(index.rglob('*'))]
    manifest={'schema_version':1,'status':'prepared','revision':revision,'url':url,
      'implementation':'bm25s==0.2.14; PyStemmer==3.0.0','parameters':{'k1':.9,'b':.4,'stopwords':'en','stemmer':'english'},
      'source':'GEPA artifact hover_program.py','files':[{'path':str(p.relative_to(ROOT)),'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in files if p.is_file()]}
    (ROOT/'reports/hover_retrieval.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'ready','files':len(manifest['files'])}))
if __name__=='__main__': main()
