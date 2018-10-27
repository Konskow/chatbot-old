import codecs
import csv
import os
import pickle
import re
import unicodedata
from typing import List, Dict

from vocabulary import Vocabulary

MAX_LENGTH = 10
MIN_COUNT = 3


def load_lines(file_name: str, fields: List[str]) -> Dict[str, Dict[str, str]]:
    lines = {}
    with open(file_name, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            line_obj = {}
            for i, field in enumerate(fields):
                line_obj[field] = values[i]
            lines[line_obj['lineID']] = line_obj
    return lines


# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
def load_conversations(file_name: str, lines: List[str], fields: List[str]) -> List[Dict[str, str]]:
    conversations = []
    with open(file_name, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            conv_obj = {}
            for i, field in enumerate(fields):
                conv_obj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            line_ids = eval(conv_obj["utteranceIDs"])
            # Reassemble lines
            conv_obj["lines"] = []
            for lineId in line_ids:
                conv_obj["lines"].append(lines[lineId])
            conversations.append(conv_obj)
    return conversations


# Extracts pairs of sentences from conversations
def extract_sentence_pairs(conversations: List[Dict[str, str]]):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            input_line = conversation["lines"][i]["text"].strip()
            target_line = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if input_line and target_line:
                qa_pairs.append([input_line, target_line])
    return qa_pairs


def convert_data(datafile: str, corpus: str):
    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    movie_lines_fields = ["lineID", "characterID", "movieID", "character", "text"]
    movie_conversations_fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    print("\nProcessing corpus...")
    lines = load_lines(os.path.join(corpus, "movie_lines.txt"), movie_lines_fields)

    print("\nLoading conversations...")
    conversations = load_conversations(os.path.join(corpus, "movie_conversations.txt"),
                                       lines,
                                       movie_conversations_fields)

    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter)
        for pair in extract_sentence_pairs(conversations):
            writer.writerow(pair)


# Lowercase, trim, and remove non-letter characters
def normalize_string(s: str) -> str:
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Read query/response pairs and return a voc object
def read_vocabularies(datafile: str, corpus_name: str) -> (Vocabulary, List[List[str]]):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    vocabulary = Vocabulary(corpus_name)
    return vocabulary, pairs


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filter_pair(p: List[List[str]]) -> List[List[str]]:
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


# Filter pairs using filterPair condition
def filter_pairs(pairs: List[List[str]]) -> List[List[str]]:
    return [pair for pair in pairs if filter_pair(pair)]


def load_prepare_data(corpus_name: str, datafile: str) -> (Vocabulary, List[List[str]]):
    print("Start preparing training data ...")
    voc, pairs = read_vocabularies(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filter_pairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.add_sentence(pair[0])
        voc.add_sentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


def unicode_to_ascii(s: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def trim_rare_words(vocabulary: Vocabulary, pairs: List[List[str]]) -> List[List[str]]:
    # Trim words used under the MIN_COUNT from the voc
    vocabulary.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in vocabulary.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in vocabulary.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


if __name__ == "__main__":
    corpus_name = "cornell movie-dialogs corpus"
    corpus = os.path.join("data", corpus_name)
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    convert_data(datafile, corpus)
    # print("\nSample lines from file:")
    # print_lines(datafile)

    voc, pairs = load_prepare_data(corpus_name, datafile)
    pairs = trim_rare_words(voc, pairs)

    pickle.dump(voc, open('data/voc.pickle', mode='wb'))
    pickle.dump(pairs, open('data/pairs.pickle', mode='wb'))
