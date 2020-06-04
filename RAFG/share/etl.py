# coding: utf-8
# create by tongshiwei on 2019/4/18
import warnings
import json
import random

import numpy as np
import mxnet as mx
from gluonnlp.data import FixedBucketSampler, PadSequence
from tqdm import tqdm

__all__ = ["etl", "pseudo_data_generation", "transform"]


def pseudo_data_generation():
    """生成伪数据流，格式和load_data中保持一致"""
    length = 20
    word_length = sorted([random.randint(1, length) for _ in range(1000)])
    char_length = sorted([i + random.randint(0, 5) for i in word_length])
    word_feature = [[random.randint(0, length) for _ in range(i)] for i in
                    word_length]
    word_radical_feature = [[random.randint(0, length) for _ in range(i)] for i
                            in word_length]
    char_feature = [[random.randint(0, length) for _ in range(i)] for i in
                    char_length]
    char_radical_feature = [[random.randint(0, length) for _ in range(i)] for i
                            in char_length]

    features = [word_feature, word_radical_feature, char_feature,
                char_radical_feature]
    labels = [random.randint(0, 32) for _ in word_length]

    return features, labels


def extract(data_src):
    word_feature = []
    word_radical_feature = []
    char_feature = []
    char_radical_feature = []
    features = [word_feature, word_radical_feature, char_feature,
                char_radical_feature]
    labels = []
    with open(data_src) as f:
        for line in tqdm(f, "loading data from %s" % data_src):
            ds = json.loads(line)
            data, label = ds['x'], ds['z']
            try:
                assert len(data[0]) == len(data[1]), "some word miss radical"
                assert len(data[2]) == len(data[3]), "some char miss radical"
            except AssertionError as e:
                warnings.warn("%s" % e)
                continue
            word_feature.append(data[0])
            word_radical_feature.append(data[1])
            char_feature.append(data[2])
            char_radical_feature.append(data[3])
            labels.append(label)

    return features, labels


def transform(raw_data, params):
    # 在这里定义数据加载方法
    batch_size = params.batch_size
    padding = params.padding
    num_buckets = params.num_buckets
    fixed_length = params.fixed_length

    features, labels = raw_data
    word_feature, word_radical_feature, char_feature, char_radical_feature = features
    batch_idxes = FixedBucketSampler(
        [len(word_f) for word_f in word_feature],
        batch_size, num_buckets=num_buckets
    )
    batch = []
    for batch_idx in batch_idxes:
        batch_features = [[] for _ in range(len(features))]
        batch_labels = []
        for idx in batch_idx:
            for i, feature in enumerate(batch_features):
                batch_features[i].append(features[i][idx])
            batch_labels.append(labels[idx])
        batch_data = []
        word_mask = []
        char_mask = []
        for i, feature in enumerate(batch_features):
            max_len = max(
                [len(fea) for fea in feature]
            ) if not fixed_length else fixed_length
            padder = PadSequence(max_len, pad_val=padding)
            feature, mask = zip(*[(padder(fea), len(fea)) for fea in feature])
            if i == 0:
                word_mask = mask
            elif i == 2:
                char_mask = mask
            batch_data.append(mx.nd.array(feature))
        batch_data.append(mx.nd.array(word_mask))
        batch_data.append(mx.nd.array(char_mask))
        batch_data.append(mx.nd.array(batch_labels, dtype=np.int))
        batch.append(batch_data)
    return batch[::-1]


def etl(filename, cfg):
    raw_data = extract(filename)
    return transform(raw_data, cfg)


if __name__ == '__main__':
    for data in tqdm(extract("../../data/data/train")):
        pass
