import json
import os
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
import _pickle as pickle
import numpy as np
from nltk.corpus import stopwords
from string import punctuation
from lemmatization import lemmatize
from tqdm import tqdm
from valence import assign_valence
from dependency_parsing import shift_valence, dependency_parse


def tokenize_review(review):
    tokens = word_tokenize(review.lower())
    return tokens


def parse_json_review(review):
    review_json = json.loads(review)
    review_id = review_json['review_id']
    review_text = tokenize_review(review_json['text'])
    # return {'Review_ID': review_id, 'Text': review_text}
    return review_id, review_text


def filter_reviews(file_name):
    frequencies = pd.read_csv('data/user_review_frequencies.csv', sep=',', index_col=0)
    lengths = pd.read_csv('data/reviews_length.csv', sep=',', index_col=0)
    p_25 = np.percentile(lengths.get_values(), 25)
    p_75 = np.percentile(lengths.get_values(), 75)
    total = 0
    filtered_50_500 = 0
    filtered_10_500 = 0
    with open(file_name, 'r', encoding='utf-8') as doc_r, \
            open('data/yelp_reviews_filtered_50_500.json', 'w+', encoding='utf-8') as doc_w1, \
            open('data/yelp_reviews_filtered_10_500.json', 'w+', encoding='utf-8') as doc_w2:
        line = doc_r.readline()
        i = 0
        while line != '':
            i += 1
            if i % 100000 == 0:
                print(str(i))
            review = json.loads(line)
            total += 1
            f = frequencies.loc[review['user_id']].values[0]
            l = lengths.loc[review['review_id']].values[0]
            if f in range(50, 500) and l in range(int(p_25), int(p_75)):
                filtered_50_500 += 1
                doc_w1.write(line)
            if f in range(10, 500) and l in range(int(p_25), int(p_75)):
                filtered_10_500 += 1
                doc_w2.write(line)
            line = doc_r.readline()
    print('Total: ' + str(total))
    print('Filtered 50-500: ' + str(filtered_50_500))
    print('Percentage 50-500: ' + str(filtered_50_500 * 100.0 / total))
    print('Filtered 10-500: ' + str(filtered_10_500))
    print('Percentage 10-500: ' + str(filtered_10_500 * 100.0 / total))


def tokenize_reviews(r1, r2):
    reviews = dict()
    with open(f'data/yelp_reviews_filtered_{r1}_{r2}.json', 'r', encoding='utf-8') as doc_r:
        line = doc_r.readline()
        i = 0
        while line != '':
            i += 1
            if i % 100000 == 0:
                print(str(i))
            review_id, review_text = parse_json_review(line)
            reviews[review_id] = review_text
            line = doc_r.readline()
    with open(f'data/yelp_reviews_tokens_{r1}_{r2}.pkl', 'wb') as doc_w:
        pickle.dump(reviews, doc_w)


def lemmatize_reviews(r1, r2):
    with open(f'data/yelp_reviews_tokens_{r1}_{r2}.pkl', 'rb') as doc_r:
        data = pickle.load(doc_r)
    reviews = dict()
    for _, review_id in zip(tqdm(list(range(len(list(data.keys()))))), list(data.keys())):
        review_tokens = data[review_id]
        review_lemmas = lemmatize(review_tokens)
        reviews[review_id] = review_lemmas
    with open(f'data/yelp_reviews_lemmas_{r1}_{r2}.pkl', 'wb') as doc_w:
        pickle.dump(reviews, doc_w)


def create_reviews_subset(file_name):
    reviews = list()
    s = [0, 0, 0, 0, 0]
    with open(file_name, 'r', encoding='utf-8') as doc:
        line = doc.readline()
        while line != '':
            review = json.loads(line)
            stars = review['stars']
            if s[int(stars) - 1] < 500:
                s[int(stars) - 1] += 1
                review_id = review['review_id']
                review_text = review['text']
                reviews.append({'Review_ID': review_id , 'Text': review_text, 'Stars': stars})
            line = doc.readline()
    with open('data/yelp_reviews_subset.json', 'w+', encoding='utf-8') as doc:
        json.dump(reviews, doc)


def filter_vocabulary(vocab, frequencies):
    allowed_tokens = [f[0] for f in frequencies if f[1] > 15]
    return [w for w in vocab if w in allowed_tokens and w not in stopwords.words('english') and w not in punctuation]


def create_vocabulary(r1, r2):
    with open(f'data/yelp_reviews_lemmas_{r1}_{r2}.pkl', 'rb') as doc_r:
        data = pickle.load(doc_r)
    vocabulary = list()
    for _, review_id in zip(tqdm(list(range(len(list(data.keys()))))), list(data.keys())):
        review_lemmas = data[review_id]
        vocabulary.extend(review_lemmas)
    pd.DataFrame(list(set(vocabulary))).to_csv(f'data/vocabulary_full_{str(r1)}_{str(r2)}.csv')
    frequencies = sorted(Counter(list(vocabulary)).items(), key=lambda x: x[1], reverse=True)
    with open(f'data/vocabulary_full_frequencies_{str(r1)}_{str(r2)}.pkl', 'wb') as f:
        pickle.dump(frequencies, f)
    filtered_vocabulary = filter_vocabulary(list(set(vocabulary)), frequencies)
    pd.DataFrame(list(set(filtered_vocabulary))).to_csv(f'data/vocabulary_{str(r1)}_{str(r2)}.csv')


def assign_valence_vocabulary(r1, r2):
    vocabulary = pd.read_csv(f'data/vocabulary_{str(r1)}_{str(r2)}.csv', index_col=[0]).get_values().flatten()
    valences_nrc = assign_valence(vocabulary, ['afinn', 'nrc-hashtag', 'nrc-vad'])
    pd.DataFrame().from_dict(valences_nrc, orient='index').to_csv(f'data/vocabulary_{str(r1)}_{str(r2)}_val_nrc.csv')
    valences_yelp = assign_valence(vocabulary, ['yelp-sentiment'])
    pd.DataFrame().from_dict(valences_yelp, orient='index').to_csv(f'data/vocabulary_{str(r1)}_{str(r2)}_val_yelp.csv')


def assign_valence_reviews(r1, r2):
    with open(f'data/yelp_reviews_lemmas_{r1}_{r2}.pkl', 'rb') as doc_r:
        data = pickle.load(doc_r)
    val_nrc = pd.read_csv(f'data/vocabulary_{str(r1)}_{str(r2)}_val_nrc.csv', index_col=[0])
    val_y = pd.read_csv(f'data/vocabulary_{str(r1)}_{str(r2)}_val_yelp.csv', index_col=[0])
    lemmas_nrc = dict()
    values_nrc = dict()
    lemmas_yelp = dict()
    values_yelp = dict()
    for _, review_id in zip(tqdm(list(range(len(list(data.keys()))))), list(data.keys())):
        lemmas = list(set(data[review_id]))
        lemmas_list_nrc = sorted([(lemma, round(val_nrc.loc[lemma][0], 5)) for lemma in lemmas
                                  if lemma in val_nrc.index], key=lambda x: x[1], reverse=True)
        lemmas_nrc[review_id] = [x[0] for x in lemmas_list_nrc]
        values_nrc[review_id] = [x[1] for x in lemmas_list_nrc]
        lemmas_list_yelp = sorted([(lemma, round(val_y.loc[lemma][0], 5)) for lemma in lemmas
                                   if lemma in val_y.index], key=lambda x: x[1], reverse=True)
        lemmas_yelp[review_id] = [x[0] for x in lemmas_list_yelp]
        values_yelp[review_id] = [x[1] for x in lemmas_list_yelp]
    pd.DataFrame().from_dict(lemmas_nrc, orient='index').to_csv(
        f'data/yelp_reviews_lemmas_val_nrc_words_{r1}_{r2}.csv')
    pd.DataFrame().from_dict(values_nrc, orient='index').to_csv(
        f'data/yelp_reviews_lemmas_val_nrc_values_{r1}_{r2}.csv')
    pd.DataFrame().from_dict(lemmas_yelp, orient='index').to_csv(
        f'data/yelp_reviews_lemmas_val_yelp_words_{r1}_{r2}.csv')
    pd.DataFrame().from_dict(values_yelp, orient='index').to_csv(
        f'data/yelp_reviews_lemmas_val_yelp_values_{r1}_{r2}.csv')


def parse_dependencies():
    if os.path.exists('data/yelp_reviews_dependencies.pkl'):
        with open('data/yelp_reviews_dependencies.pkl', 'rb') as doc_r:
            dependencies = pickle.load(doc_r)
    else:
        dependencies = dict()
    with open('data/yelp_reviews_filtered_50_500.json', 'r', encoding='utf-8') as doc_r:
        line = doc_r.readline()
        i = 0
        while line != '':
            i += 1
            if i % 10 == 0:
                print(str(i))
            review = json.loads(line)
            review_id = review['review_id']
            if review_id not in dependencies.keys():
                review_text = review['text']
                try:
                    rel = dependency_parse(review_text.replace('/', ' '))
                except AssertionError:
                    rel = []
                    print(review_text)
                    with open('data/yelp_reviews_dependencies.pkl', 'wb') as doc_w:
                        pickle.dump(dependencies, doc_w)
                dependencies[review_id] = rel
                if i % 2000 == 0:
                    with open(f'data/yelp_reviews_dependencies_{i}.pkl', 'wb') as doc_w:
                        pickle.dump(dependencies, doc_w)
            line = doc_r.readline()
    with open('data/yelp_reviews_filtered_10_500.json', 'r', encoding='utf-8') as doc_r:
        line = doc_r.readline()
        i = 0
        while line != '':
            i += 1
            if i % 100 == 0:
                print(str(i))
            review = json.loads(line)
            review_id = review['review_id']
            if review_id not in dependencies.keys():
                review_text = review['text']
                try:
                    rel = dependency_parse(review_text.replace('/', ' '))
                except AssertionError:
                    rel = []
                    print(review_text)
                    with open('data/yelp_reviews_dependencies.pkl', 'wb') as doc_w:
                        pickle.dump(dependencies, doc_w)
                dependencies[review_id] = rel
                if i % 3000 == 0:
                    with open(f'data/yelp_reviews_dependencies_{i}.pkl', 'wb') as doc_w:
                        pickle.dump(dependencies, doc_w)
            line = doc_r.readline()
    with open('data/yelp_reviews_dependencies.pkl', 'wb') as doc_w:
        pickle.dump(dependencies, doc_w)


def context_shift(r1, r2):
    lemmas_nrc = pd.read_table(f'data/yelp_reviews_lemmas_val_nrc_words_{r1}_{r2}.csv', sep=',',
                               index_col=[0], keep_default_na=False, na_values=['nan']).to_dict(orient='index')
    for key, val in list(lemmas_nrc.items()):
        lemmas_nrc[key] = {k: v for k, v in val.items() if v is not ''}
    values_nrc = pd.read_table(f'data/yelp_reviews_lemmas_val_nrc_values_{r1}_{r2}.csv', sep=',',
                               index_col=[0], keep_default_na=False, na_values=['nan']).to_dict(orient='index')
    for key, val in list(values_nrc.items()):
        values_nrc[key] = {k: v for k, v in val.items() if v is not ''}
    lemmas_yelp = pd.read_table(f'data/yelp_reviews_lemmas_val_yelp_words_{r1}_{r2}.csv', sep=',',
                               index_col=[0], keep_default_na=False, na_values=['nan']).to_dict(orient='index')
    for key, val in list(lemmas_yelp.items()):
        lemmas_yelp[key] = {k: v for k, v in val.items() if v is not ''}
    values_yelp = pd.read_table(f'data/yelp_reviews_lemmas_val_yelp_values_{r1}_{r2}.csv', sep=',',
                               index_col=[0], keep_default_na=False, na_values=['nan']).to_dict(orient='index')
    for key, val in list(values_yelp.items()):
        values_yelp[key] = {k: v for k, v in val.items() if v is not ''}
    with open('data/yelp_reviews_dependencies_6000.pkl', 'rb') as doc_r:
        dependencies = pickle.load(doc_r)
    with open(f'data/yelp_reviews_filtered_{r1}_{r2}.json', 'r', encoding='utf-8') as doc_r:
        line = doc_r.readline()
        i = 0
        while line != '':
            i += 1
            if i % 50000 == 0:
                print(str(i))
            review = json.loads(line)
            review_id = review['review_id']
            if review_id in lemmas_nrc.keys() and review_id in dependencies.keys() and len(dependencies[review_id]) > 0:
                review_text = review['text']
                lemmas = list(lemmas_nrc[review_id].values())
                values = [float(v) for v in list(values_nrc[review_id].values())]
                rel = dependencies[review_id]
                val_shift, rel = shift_valence(review_text.replace('/', ' '), lemmas, values, rel)
                # lemmas_list_nrc = sorted([(l, v) for l, v in zip(lemmas, val_shift)], key=lambda x: x[1], reverse=True)
                # lemmas_nrc[review_id] = [x[0] for x in lemmas_list_nrc]
                # values_nrc[review_id] = [x[1] for x in lemmas_list_nrc]
                values_nrc[review_id] = val_shift
                lemmas = list(lemmas_yelp[review_id].values())
                values = [float(v) for v in list(values_yelp[review_id].values())]
                val_shift, rel = shift_valence(review_text.replace('/', ' '), lemmas, values, rel)
                # lemmas_list_yelp = sorted([(l, v) for l, v in zip(lemmas, val_shift)], key=lambda x: x[1], reverse=True)
                # lemmas_yelp[review_id] = [x[0] for x in lemmas_list_yelp]
                # values_yelp[review_id] = [x[1] for x in lemmas_list_yelp]
                values_yelp[review_id] = val_shift
            line = doc_r.readline()
    # pd.DataFrame().from_dict(lemmas_nrc, orient='index').to_csv(
    #     f'data/yelp_reviews_lemmas_val_shifted_nrc_words_{r1}_{r2}.csv')
    pd.DataFrame().from_dict(values_nrc, orient='index').to_csv(
        f'data/yelp_reviews_lemmas_val_shifted_nrc_values_{r1}_{r2}.csv')
    # pd.DataFrame().from_dict(lemmas_yelp, orient='index').to_csv(
    #     f'data/yelp_reviews_lemmas_val_shifted_yelp_words_{r1}_{r2}.csv')
    pd.DataFrame().from_dict(values_yelp, orient='index').to_csv(
        f'data/yelp_reviews_lemmas_val_shifted_yelp_values_{r1}_{r2}.csv')


def create_user_review_frequencies(file_name):
    frequencies = dict()
    with open(file_name, 'r', encoding='utf-8') as doc:
        line = doc.readline()
        i = 0
        while line != '':
            i += 1
            if i % 100000 == 0:
                print(str(i))
            review = json.loads(line)
            user_id = review['user_id']
            if user_id in frequencies.keys():
                frequencies[user_id] += 1
            else:
                frequencies[user_id] = 1
            line = doc.readline()
    pd.DataFrame().from_dict(frequencies, orient='index').to_csv('data/user_review_frequencies.csv')


def calculate_reviews_length(file_name):
    lengths = dict()
    with open(file_name, 'r', encoding='utf-8') as doc:
        line = doc.readline()
        i = 0
        while line != '':
            i += 1
            if i % 100000 == 0:
                print(str(i))
            review = json.loads(line)
            lengths[review['review_id']] = len(tokenize_review(review['text']))
            line = doc.readline()
    pd.DataFrame().from_dict(lengths, orient='index').to_csv('data/reviews_length.csv')


if __name__ == '__main__':
    # create_reviews_subset('data/yelp_reviews.json')
    # create_user_review_frequencies('data/yelp_reviews.json')
    # calculate_reviews_length('data/yelp_reviews.json')
    # filter_reviews('data/yelp_reviews.json')
    # tokenize_reviews(50, 500)
    # tokenize_reviews(10, 500)
    # lemmatize_reviews(50, 500)
    # lemmatize_reviews(10, 500)
    # create_vocabulary(50, 500)
    # create_vocabulary(10, 500)
    # assign_valence_vocabulary(50, 500)
    # assign_valence_vocabulary(10, 500)
    # assign_valence_reviews(50, 500)
    # assign_valence_reviews(10, 500)
    # parse_dependencies()
    context_shift(50, 500)
    context_shift(10, 500)
