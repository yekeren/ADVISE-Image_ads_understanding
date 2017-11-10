import sys
import struct
import numpy as np
import pdb

def _cosine(v1, v2):
    ''' Compute cosine distance between two vectors.
    Args:
        v1: The first vector.
        v2: The second vector.
    Returns:
        dist: The cosine distance.
    '''
    v1v2 = float(np.multiply(v1, v2).sum())
    v1v1 = float(np.multiply(v1, v1).sum())
    v2v2 = float(np.multiply(v2, v2).sum())

    return float(v1v2 / np.sqrt(v1v1 * v2v2))

class EmbeddingConverter(object):
    def __init__(self):
        self._words = 0
        self._size = 0

        self._lines = []
        self._vocab = []
        self._embedding = None
        self._index = {}

    def saveEmbedding(self, filename):
        ''' Save the embeddings to npz file.
        Args:
            filename
        '''
        with open(filename, 'wb') as fp:
            np.save(fp, self._embedding)
        print >> sys.stderr, 'Save embedding to %s' % (filename)

    def saveVocab(self, filename):
        ''' Save vocab to text file.
        Args:
            filename
        '''
        with open(filename, 'wb') as fp:
            for w in self._vocab:
                fp.write(w + '\n')
        print >> sys.stderr, 'Save vocab to %s' % (filename)

    def save(self, filename):
        ''' Save file using the original format.
        Args:
            filename
        '''
        with open(filename, 'wb') as fp:
            for line in self._lines:
                fp.write(line)
        print >> sys.stderr, 'Save origin to %s' % (filename)

    def getDims(self):
        ''' Get the embeddingdimensions.
        Returns:
            dims
        '''
        return self._size

    def getVecByWord(self, word):
        ''' Get vector by word.
        Args:
            word
        '''
        if not word in self._index:
            return None
        return self._embedding[self._index[word]]

    def getSynonymsByWord(self, word, top_k=40):
        ''' Get synonyms by word.
        Args:
            word
            top_k
        Returns:
            synonyms: A list of synonyms.
        '''
        dist = []
        emb = self.getVecByWord(word)
        for i in xrange(self._words):
            dist.append((i, _cosine(emb, self._embedding[i])))
        dist = sorted(dist, lambda x, y: - cmp(x[1], y[1]))[:top_k]
        dist = map(lambda x: (self._vocab[x[0]], x[1]), dist)
        return dist

class Word2VecConverter(EmbeddingConverter):
    def _decodeLine(self, fp):
        ''' Decode a line in the file.
        Args:
            fp: The file descriptor.
        '''
        # Read the word from the file.
        w = ''
        while True:
            ch = fp.read(1)
            if ch == ' ': break
            w += ch
        line = fp.read(4 * self._size)

        # Read the associated feature.
        vec = np.zeros(self._size, dtype=np.float32)
        for i in xrange(self._size):
            vec[i] = struct.unpack_from('<f', line, offset=4 * i)[0]
        return w, vec

    def load(self, filename):
        ''' Load a pre-trained word2vec file.
        Args:
            filename: The pre-trained word2vec model.
        '''
        self._vocab = []
        with open(filename, 'rb') as fp:
            self._words, self._size = map(int, fp.readline().split())
            self._embedding = np.zeros(
                    (self._words, self._size), dtype=np.float32)

            for b in xrange(self._words):
                w, vec = self._decodeLine(fp)
                self._vocab.append(w)
                self._embedding[b] = vec

                if b % 10000 == 0:
                    print >> sys.stderr, 'Load %s words' % (b)

        for i, w in enumerate(self._vocab):
            self._index[w] = i

        print >> sys.stderr, 'Load data from %s' %(filename)

class GloveConverter(EmbeddingConverter):
    def load(self, filename, words=None):
        ''' Load a pre-trained word2vec file.
        Args:
            filename: The pre-trained word2vec model.
        '''
        with open(filename, 'r') as fp:
            lines = fp.readlines()

        self._words = len(lines)
        self._size = len(lines[0].strip('\n').split()) - 1

        # Load data from file
        self._vocab = []
        vecs = []
        for b, line in enumerate(lines):
            items = line.strip('\n').split()
            w, vec = items[0], map(float, items[1:])
            assert len(vec) == self._size

            if words is None or w in words:
                self._lines.append(line)
                self._vocab.append(w)
                vecs.append(np.array([vec], dtype=np.float32))

            if b % 1927 == 0:
                print >> sys.stderr, 'Loading %s/%s\r' % (b + 1, len(lines)),
        print >> sys.stderr, 'Loading %s/%s' % (len(lines), len(lines))

        self._words = len(self._vocab)
        self._embedding = np.concatenate(vecs, 0)

        for i, w in enumerate(self._vocab):
            self._index[w] = i

        print >> sys.stderr, 'Loaded %s glove embeddings' % (len(self._vocab))
        return self._vocab, self._embedding
