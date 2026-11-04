"""Chen-Junbao/FedCCFA 647d8b0 の式の抜粋（外部cloneに依存しない）。

MIT License
Copyright (c) 2024 Junbao Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import torch
from sklearn.cluster import DBSCAN


def madd(vecs):
    """entities/FedCCFA.py:335-349。"""
    result = np.zeros((len(vecs), len(vecs)))

    def cosine(a, b):
        return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            result[i, j] = result[j, i] = sum(
                abs(cosine(vecs[i], vecs[z]) - cosine(vecs[j], vecs[z]))
                for z in range(len(vecs))
                if z not in (i, j)
            ) / (len(vecs) - 2)
    return result


def groups(vecs, eps):
    """entities/FedCCFA.py:353-388。2 client以上だけを返す。"""
    labels = DBSCAN(eps=eps, min_samples=1, metric="precomputed").fit(madd(vecs)).labels_
    return {
        frozenset(np.where(labels == label)[0])
        for label in set(labels)
        if np.sum(labels == label) > 1
    }


def mean(values):
    """entities/FedCCFA.py:420-445,448-476。等重みの行とアンカーの平均。"""
    result = torch.zeros_like(values[0])
    for value in values:
        result += value
    return result / len(values)


def representation(values, counts):
    """entities/FedCCFA.py:277-294。件数の重みで表現部を平均。"""
    result = torch.zeros_like(values[0])
    for value, count in zip(values, counts):
        result += value * count
    return result / sum(counts)


def weight(counts, gamma):
    """entities/FedCCFA.py:31-42。全クラスが存在する入力に限る。"""
    p = np.array(counts) / sum(counts)
    return -np.sum(p * np.log(p)) / gamma


def alignment(features, anchors, labels, temperature):
    """entities/FedCCFA.py:131-149。CLの式。"""
    protos = torch.stack(anchors).repeat(len(labels), 1, 1)
    logits = torch.nn.CosineSimilarity(dim=2)(features.unsqueeze(1), protos)
    return torch.nn.functional.cross_entropy(logits / temperature, labels)


def balanced_count(counts):
    """entities/FedCCFA.py:188-201。クラスごとの抽出件数。"""
    return min(min(counts), 5)
