# cs229-finalproject
This repository contains stuff about CS 229 final project

## Features
#### BOW(Bag-Of-Word) Features
To generate BOW features, run the following command in the terminal
```
$ python Features/token_features.py 1 <body_raw_file> <headline_raw_file>
``` 

Output files are in the "Feature/BOW vectors" folder, includes:

- Dictionary:  dict.dict
- Term Frequency Corpora
    
    - A corpus contains all headlines: tf_headline_corpus.mm
    - A corpus contains all article bodies: tf_body_corpus.mm
    - Each article body/headline is a list of (word_id, word_frequency) tuples
    - Each corpus is a list of body/headline (a list of a list of 2D tuples)
    
- Tfidf Corpora

    - A corpus contains all headlines: tfidf_headline_corpus.mm
    - A corpus contains all article bodies: tfidf_body_corpus.mm
    - Each article body/headline is a list of (word_id, tfidf) tuples
    - Each corpus is a list of body/headline (a list of a list of 2D tuples)
    
- Stance: stance_output.txt  

**Headline with index x corresponds to the article body with index x, and the stance with index x.**

(Current files in the BOW vector fold are generated according to 9 training items for testing purpose.)

To load a corpus:
```python
from gensim import corpora
corpus = corpora.MmCorpus('filename')
```
#### N-grams Features
To generate N-grams features:  (n denotes the number of words in a token)
```
$ python Features/token_features.py n <body_raw_file> <headline_raw_file>
``` 
For example, to generate bigram features, run:
```
$ python Features/token_features.py 2 <body_raw_file> <headline_raw_file>
``` 
Output files will be in a folder called "$n$grams vectors". 

Output files are the same as the BOW features
