import torch
from collections import defaultdict
import numpy as np


def group_by_first_token(texts, tokenizer):
    # Group a list of texts based on the first token in each text.

    seqs = [tokenizer.encode(x, add_special_tokens=False) for x in texts]
    grouped = defaultdict(list)
    for seq in seqs:
        grouped[seq[0]].append(seq)
    return grouped

def tokens_to_string(token_ids, v):
        # Merge subword tokens into whole words
        indices = []
        for i, token_id in enumerate(token_ids):
            token_text = v[token_id]
            if token_text.startswith("##"):
                indices.append(i)
            else:
                if indices:
                    toks = [v[token_ids[t]] for t in indices]
                    word = "".join([toks[0]] + [t[2:] for t in toks[1:]])
                    yield indices, word
                indices = [i]

def bpe_tokenize(bpe_tokenizer, sentence):
    # Tokenize a sentence using a given Byte Pair Encoding (BPE) tokenizer.

    sent_bpe_tokens = []
    sent_bpe_offsets = []
    for token in sentence:
        # Tokenize the token's text using the BPE tokenizer.
        token_bpes = bpe_tokenizer.tokenize(token.text)

        # Store the offsets for each subword token.
        sent_bpe_offsets += [(token.begin, token.end) for _ in range(len(token_bpes))]
        sent_bpe_tokens += token_bpes

    return sent_bpe_tokens, sent_bpe_offsets

def nlargest_indexes(arr, n_top):
    # Find the indices of the top n elements in an array 'arr'.

    arr_ids = np.argpartition(arr, -n_top)[-n_top:]
    sel_arr = arr[arr_ids]
    top_ids = arr_ids[np.argsort(-sel_arr)]
    return top_ids


def remove_masked_token_subwords(masked_position, bpe_tokens, bpe_offsets):
    """
    If the masked token has been tokenied into multiple subwords: like dieting-->diet and ##ing
    keep the first subword and remove others.
    """

    # Check if the masked token has been tokenized into multiple subwords.
    if len(masked_position[1]) > 1:
        # Get the indexes to delete, which correspond to subwords after the first.
        indexes_to_del = masked_position[1][1:]

        # Delete the subword tokens from 'bpe_tokens' and their corresponding offsets.
        del bpe_tokens[masked_position[0]][indexes_to_del[0] : indexes_to_del[-1] + 1]
        del bpe_offsets[masked_position[0]][indexes_to_del[0] : indexes_to_del[-1] + 1]

    # Update 'masked_position' to represent the first subword only.
    masked_position = (
        masked_position[0],
        masked_position[1][0],
    )

    return masked_position, bpe_tokens, bpe_offsets


def merge_sorted_results(
    objects_left, scores_left, objects_right, scores_right, max_elems
):
    # Merge and sort two lists of results based on their scores

    result_objects = []
    result_scores = []

    j = 0
    i = 0
    while True:
        # Check if the maximum number of elements has been reached.
        if len(result_scores) == max_elems:
            break

        # Check if the left list has been fully processed.
        if i == len(scores_left):
            # Append remaining elements from the right list to the result.
            result_objects += objects_right[j : j + max_elems - len(result_scores)]
            result_scores += scores_right[j : j + max_elems - len(result_scores)]
            break

        # Check if the right list has been fully processed.
        if j == len(scores_right):
            # Append remaining elements from the left list to the result.
            result_objects += objects_left[i : i + max_elems - len(result_scores)]
            result_scores += scores_left[i : i + max_elems - len(result_scores)]
            break

        # Compare the scores of elements from the left and right lists and append the one with the larger score
        if scores_left[i] > scores_right[j]:
            result_objects.append(objects_left[i])
            result_scores.append(scores_left[i])
            i += 1
        else:
            result_objects.append(objects_right[j])
            result_scores.append(scores_right[j])
            j += 1

    return result_objects, result_scores

def calculate_cosine_similarity(vec1, vec2):
    # Function to compute cosine similarity between two vectors
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)