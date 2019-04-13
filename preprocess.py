import json
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
import _pickle as pickle
import numpy as np


def tokenize_review(review):
    tokens = word_tokenize(review.lower())
    return tokens


def parse_json_review(review):
    review_json = json.loads(review)
    review_id = review_json['review_id']
    review_text = tokenize_review(review_json['text'])
    return {'Review_ID': review_id, 'Text': review_text}


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


def tokenize(file_name, r1, r2):
    reviews = list()
    with open(file_name, 'r', encoding='utf-8') as doc_r:
        line = doc_r.readline()
        i = 0
        while line != '':
            i += 1
            if i % 100000 == 0:
                print(str(i))
            review = parse_json_review(line)
            reviews.append(review)
            line = doc_r.readline()
    with open(f'data/yelp_reviews_tokens_{r1}_{r2}.pkl', 'wb') as doc_w:
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
    ...


def create_vocabulary(file_name, r1, r2):
    vocabulary = list()
    with open(file_name, 'r', encoding='utf-8') as doc:
        line = doc.readline()
        i = 0
        while line != '':
            i += 1
            if i % 100000 == 0:
                print(str(i))
            review = json.loads(line)
            vocabulary.extend(tokenize_review(review['text']))
            line = doc.readline()
    pd.DataFrame(list(set(vocabulary))).to_csv(f'data/vocabulary_full_{str(r1)}_{str(r2)}.csv')
    frequencies = sorted(Counter(list(vocabulary)).items(), key=lambda x: x[1], reverse=True)
    with open(f'data/vocabulary_full_frequencies_{str(r1)}_{str(r2)}.pkl', 'wb') as f:
        pickle.dump(frequencies, f)


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
    # create_vocabulary('data/yelp_reviews_filtered_50_500.json', 50, 500)
    # create_vocabulary('data/yelp_reviews_filtered_10_500.json', 10, 500)
    tokenize('data/yelp_reviews_filtered_50_500.json', 50, 500)
    tokenize('data/yelp_reviews_filtered_10_500.json', 10, 500)
