import itertools
from typing import List

import torch

from prepare_data import normalize_string
from vocabulary import Vocabulary


def indices_from_sentence(vocabulary: Vocabulary, sentence: str) -> List[int]:
    return [vocabulary.word2index[word] for word in sentence.split(' ')] + [vocabulary.word2index['EOS']]


def zero_padding(indices_batches: List[List[int]], fillvalue: int = 0) -> List[List[int]]:
    return list(itertools.zip_longest(*indices_batches, fillvalue=fillvalue))


def binary_matrix(l, value=0):
    matrix = []
    for i, seq in enumerate(l):
        matrix.append([])
        for token in seq:
            if token == value:
                matrix[i].append(0)
            else:
                matrix[i].append(1)
    return matrix


# Returns padded input sequence tensor and lengths
def input_tensor(sentences: List[str], vocabulary: Vocabulary, device: str = 'cpu') -> (torch.Tensor, torch.Tensor):
    indices_batch = [indices_from_sentence(vocabulary, sentence) for sentence in sentences]
    lengths = torch.tensor([len(indexes) for indexes in indices_batch]).to(device=device)
    padded_batch = zero_padding(indices_batch, vocabulary.word2index['PAD'])
    padded_tensor = torch.tensor(padded_batch).to(dtype=torch.long, device=device)
    return padded_tensor, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def output_tensor(sentences: List[str], vocabulary: Vocabulary, device: str = 'cpu') -> (torch.Tensor, torch.Tensor, int):
    indices_batch = [indices_from_sentence(vocabulary, sentence) for sentence in sentences]
    max_target_len = max([len(indexes) for indexes in indices_batch])
    padded_list = zero_padding(indices_batch, vocabulary.word2index['PAD'])
    mask = binary_matrix(padded_list, vocabulary.word2index['PAD'])
    mask = torch.tensor(mask).to(dtype=torch.uint8, device=device)
    padded_tensor = torch.tensor(padded_list).to(dtype=torch.long, device=device)
    return padded_tensor, mask, max_target_len


# Returns all items for a given batch of pairs
def batch_to_train_data(vocabulary: Vocabulary, pair_batch: List[List[List[str]]]) -> \
        (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = input_tensor(input_batch, vocabulary)
    output, mask, max_target_len = output_tensor(output_batch, vocabulary)
    return inp, lengths, output, mask, max_target_len
