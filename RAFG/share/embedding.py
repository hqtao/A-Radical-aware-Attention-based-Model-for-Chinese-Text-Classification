# coding: utf-8
# create by tongshiwei on 2019/4/18

import logging
import pathlib

from gluonnlp.embedding import TokenEmbedding
from longling.lib.clock import print_time
from longling.lib.parser import path_append
from mxnet import gluon

__all__ = [
    "load_embedding",
    "WRCEmbedding", "WCREmbedding", "WCEmbedding"
]

VEC_ROOT = pathlib.PurePath(__file__).parents[2] / "data" / "vec"


def load_embedding(vec_root=VEC_ROOT, logger=logging.getLogger()):
    """

    Parameters
    ----------
    vec_root: str
    logger: logging.logger

    """
    word_embedding_file = path_append(
        vec_root, "word.vec.dat", to_str=True
    )
    word_radical_embedding_file = path_append(
        vec_root, "word_radical.vec.dat", to_str=True
    )
    char_embedding_file = path_append(
        vec_root, "char.vec.dat", to_str=True
    )
    char_radical_embedding_file = path_append(
        vec_root, "char_radical.vec.dat", to_str=True
    )

    with print_time(logger=logger, task='loading embedding'):
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(4)

        p1 = pool.apply_async(TokenEmbedding.from_file,
                              args=(word_embedding_file,))
        p2 = pool.apply_async(TokenEmbedding.from_file,
                              args=(word_radical_embedding_file,))
        p3 = pool.apply_async(TokenEmbedding.from_file,
                              args=(char_embedding_file,))
        p4 = pool.apply_async(TokenEmbedding.from_file,
                              args=(char_radical_embedding_file,))

        pool.close()
        pool.join()

        word_embedding = p1.get()
        word_radical_embedding = p2.get()
        char_embedding = p3.get()
        char_radical_embedding = p4.get()

        return word_embedding, word_radical_embedding, \
               char_embedding, char_radical_embedding


class WRCEmbedding(gluon.HybridBlock):
    def __init__(self, word_embedding_size, word_radical_embedding_size,
                 char_embedding_size,
                 embedding_dim, dropout=0.5, prefix=None,
                 params=None):
        super(WRCEmbedding, self).__init__(prefix, params)
        with self.name_scope():
            self.word_embedding = gluon.nn.Embedding(
                word_embedding_size, embedding_dim
            )
            self.word_radical_embedding = gluon.nn.Embedding(
                word_radical_embedding_size, embedding_dim
            )
            self.char_embedding = gluon.nn.Embedding(
                char_embedding_size, embedding_dim
            )
            self.word_dropout = gluon.nn.Dropout(dropout)
            self.char_dropout = gluon.nn.Dropout(dropout)

    def hybrid_forward(self,
                       F, word_seq, word_radical_seq,
                       character_seq,
                       *args, **kwargs
                       ):
        word_embedding = self.word_embedding(word_seq)
        word_radical_embedding = self.word_radical_embedding(word_radical_seq)
        character_embedding = self.char_embedding(character_seq)

        word_embedding = self.word_dropout(word_embedding)
        word_radical_embedding = self.word_dropout(word_radical_embedding)
        character_embedding = self.char_dropout(character_embedding)

        return word_embedding, word_radical_embedding, character_embedding

    def set_weight(self, we, wre, ce):
        self.word_embedding.weight.set_data(we)
        self.word_radical_embedding.weight.set_data(wre)
        self.char_embedding.weight.set_data(ce)


class WCREmbedding(gluon.HybridBlock):
    def __init__(self, word_embedding_size, word_radical_embedding_size,
                 char_embedding_size, char_radical_embedding_size,
                 embedding_dim, dropout=0.5, prefix=None,
                 params=None):
        super(WCREmbedding, self).__init__(prefix, params)
        with self.name_scope():
            self.word_embedding = gluon.nn.Embedding(
                word_embedding_size, embedding_dim
            )
            self.word_radical_embedding = gluon.nn.Embedding(
                word_radical_embedding_size, embedding_dim
            )
            self.char_embedding = gluon.nn.Embedding(
                char_embedding_size, embedding_dim
            )
            self.char_radical_embedding = gluon.nn.Embedding(
                char_radical_embedding_size, embedding_dim
            )
            self.word_dropout = gluon.nn.Dropout(dropout)
            self.char_dropout = gluon.nn.Dropout(dropout)

    def hybrid_forward(self,
                       F, word_seq, word_radical_seq,
                       character_seq, character_radical_seq,
                       *args, **kwargs
                       ):
        word_embedding = self.word_embedding(word_seq)
        word_radical_embedding = self.word_radical_embedding(word_radical_seq)
        character_embedding = self.char_embedding(character_seq)
        character_radical_embedding = self.char_radical_embedding(
            character_radical_seq
        )

        word_embedding = self.word_dropout(word_embedding)
        word_radical_embedding = self.word_dropout(word_radical_embedding)
        character_embedding = self.char_dropout(character_embedding)
        character_radical_embedding = self.char_dropout(
            character_radical_embedding
        )

        return word_embedding, word_radical_embedding, \
               character_embedding, character_radical_embedding

    def set_weight(self, we, wre, ce, cre):
        self.word_embedding.weight.set_data(we)
        self.word_radical_embedding.weight.set_data(wre)
        self.char_embedding.weight.set_data(ce)
        self.char_radical_embedding.weight.set_data(cre)


class WCEmbedding(gluon.HybridBlock):
    def __init__(self, word_embedding_size,
                 char_embedding_size, embedding_dim, dropout=0.5, prefix=None,
                 params=None):
        super(WCEmbedding, self).__init__(prefix, params)
        with self.name_scope():
            self.word_embedding = gluon.nn.Embedding(
                word_embedding_size, embedding_dim
            )
            self.char_embedding = gluon.nn.Embedding(
                char_embedding_size, embedding_dim
            )
            self.word_dropout = gluon.nn.Dropout(dropout)
            self.char_dropout = gluon.nn.Dropout(dropout)

    def hybrid_forward(self, F, word_seq, character_seq, *args, **kwargs):
        word_embedding = self.word_embedding(word_seq)
        character_embedding = self.char_embedding(character_seq)

        word_embedding = self.word_dropout(word_embedding)
        character_embedding = self.char_dropout(character_embedding)

        return word_embedding, character_embedding

    def set_weight(self, we, ce):
        self.word_embedding.weight.set_data(we)
        self.char_embedding.weight.set_data(ce)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    load_embedding()
