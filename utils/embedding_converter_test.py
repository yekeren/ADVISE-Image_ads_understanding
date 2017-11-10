import os
import sys
import numpy as np
import tensorflow as tf

from embedding_converter import *

class EmbeddingConverterTest(tf.test.TestCase):
    def testWord2VecConverter(self):
        w2v = Word2VecConverter()
        w2v.load('data/GoogleNews-vectors-negative300.bin')
        results = w2v.getSynonymsByWord('king')
        for r in results:
            print r
        w2v.saveVocab('data/vocab.google_news.txt')
        w2v.saveEmbedding('data/embedding.google_news.npz')

    def testGloveConverter(self):
        glove = GloveConverter()
        glove.load('conf/glove.6B.50d.txt')
        results = glove.getSynonymsByWord('king')
        for r in results:
            print r
        glove.saveVocab('models/vocab.glove_6b_50d.txt')
        glove.saveEmbedding('models/embedding.glove_6b_50d.npz')

if __name__ == '__main__':
    tf.test.main()
