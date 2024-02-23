#Author: Sinan Parmar
#Date: 2024-02-21

import string
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import normalize


def extract_keyword(texts, pos_model_name='en_core_web_sm',embedding_model_name='sentence-transformers/allenai-specter', device='cpu', batch_size=128, n_process=1):

    if device != 'cpu':
        gpu = True
    else:
        gpu = False

    nlp = load_spacy_model(pos_model_name, gpu = gpu)

    if type(texts) == list:

        nostop_noun_chunks = extract_noun_chunks(texts, nlp, batch_size)
        noun_vecs = get_noun_vecs(nostop_noun_chunks, embedding_model_name, device, batch_size)
        text_vecs = get_text_vecs(texts, embedding_model_name, device, batch_size)

        text_vectors_norm = normalize(text_vecs)
        word_vectors_norm = normalize(noun_vecs)

        dotp = np.dot(word_vectors_norm, text_vectors_norm.T)
        dotp_mean = np.mean(dotp, axis=1)

        # Sort dotp_mean in descending order and get the indices
        sorted_indices = np.argsort(dotp_mean)[::-1]

        # Initialize an empty dictionary to store the top 5 unique keywords
        top_5_keywords = {}

        # Iterate over the sorted indices
        for i in sorted_indices:
            # If the keyword is not already in the dictionary, add it
            if nostop_noun_chunks[i] not in top_5_keywords:
                top_5_keywords[nostop_noun_chunks[i]] = dotp_mean[i]
            # If we have already found 5 unique keywords, break the loop
            if len(top_5_keywords) == 5:
                break

        return top_5_keywords

    elif type(texts) == dict:

        topic_nouns = {}
        for topic in texts.keys():
            topic_texts = texts[topic]
            nostop_noun_chunks = extract_noun_chunks(topic_texts, nlp, batch_size, n_process)
            topic_nouns[topic] = nostop_noun_chunks

        topic_nouns = tfidf_weight(topic_nouns)

        topic_keywords = {}
        for topic in topic_nouns.keys():
            noun_list = list(topic_nouns[topic].keys())
            noun_vecs = get_noun_vecs(noun_list, embedding_model_name, device, batch_size)
            text_vecs = get_text_vecs(texts[topic], embedding_model_name, device, batch_size)

            text_vectors_norm = normalize(text_vecs)
            word_vectors_norm = normalize(noun_vecs)

            dotp = np.dot(word_vectors_norm, text_vectors_norm.T)
            dotp_mean = np.mean(dotp, axis=1)
            w_dotp_mean =  np.multiply(list(topic_nouns[topic].values()),dotp_mean)

            sorted_indices = np.argsort(w_dotp_mean)[::-1]

            top_5_keywords = {}
            for i in sorted_indices:
                if noun_list[i] not in top_5_keywords:
                    top_5_keywords[noun_list[i]] = w_dotp_mean[i]
                if len(top_5_keywords) == 5:
                    break

            topic_keywords[topic] = top_5_keywords

        return topic_keywords


def get_text_vecs(texts, model_name, device='cpu', batch_size=128):
    model = SentenceTransformer(model_name, device=device)
    text_vecs = model.encode(texts, batch_size=batch_size)
    return text_vecs


def get_noun_vecs(noun_chunks, model_name, device='cpu', batch_size=128):
    model = SentenceTransformer(model_name, device=device)
    noun_vecs = model.encode(noun_chunks, batch_size=batch_size)
    return noun_vecs


def extract_noun_chunks(texts, nlp, batch_size=128, n_process=1):
    nostop_noun_chunks = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        for noun_chunk in doc.noun_chunks:
            text_list = [chunk.text for chunk in noun_chunk if not chunk.is_stop]
            text = " ".join(text_list).lower().strip()
            nostop_noun_chunks.append(text)

    return list(set(nostop_noun_chunks))

def tfidf_weight(topic_nouns):

    done_nouns = {}
    for topic_i in topic_nouns:
        topic_idf_dict = {}

        for noun in topic_nouns[topic_i]:

            if noun in done_nouns:
                topic_idf_dict[noun] = done_nouns[noun]
                continue

            noun_freq = 1
            for topic_j in topic_nouns:
                if topic_i != topic_j:
                    if noun in topic_nouns[topic_j]:
                        noun_freq += 1
            topic_idf_dict[noun] = 1 + np.log10(2/noun_freq)
            done_nouns.update({noun:1 + np.log10(2/noun_freq)})
        topic_nouns[topic_i] = topic_idf_dict

    return topic_nouns

def remove_punctuation(text_list):
    table = str.maketrans('', '', string.punctuation)
    return [text.translate(table) for text in text_list]


def load_spacy_model(model_name, gpu=False):
    if gpu:
        spacy.require_gpu()
    nlp = spacy.load(model_name)
    return nlp