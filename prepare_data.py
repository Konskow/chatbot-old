import codecs
import csv
import os
from typing import List, Dict


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

    # print_lines(os.path.join(corpus, "movie_lines.txt"))

    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    print("\nProcessing corpus...")
    lines = load_lines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)

    print("\nLoading conversations...")
    conversations = load_conversations(os.path.join(corpus, "movie_conversations.txt"),
                                       lines,
                                       MOVIE_CONVERSATIONS_FIELDS)

    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter)
        for pair in extract_sentence_pairs(conversations):
            writer.writerow(pair)
