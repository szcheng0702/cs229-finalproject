from gensim import corpora

corpus = corpora.MmCorpus('BOW vectors/tf_body_corpus.mm')
print (corpus[0])