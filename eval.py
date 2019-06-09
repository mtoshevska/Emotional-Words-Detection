import _pickle as pickle
import pandas as pd
from tqdm import tqdm
from scipy.stats import kendalltau, pearsonr


def find_num_mutual(rel, ret):
    return len(set(rel).intersection(set(ret)))


def calculate_jaccard(rel, ret):
    intersect = find_num_mutual(rel, ret)
    union = len(rel) + len(ret)
    jaccard = intersect / (union - intersect)
    return round(jaccard, 10)


def calculate_metrics(relevant, retrieved):
    j_5 = calculate_jaccard(relevant[:5], retrieved[:5])
    j_10 = calculate_jaccard(relevant[:10], retrieved[:10])
    j_15 = calculate_jaccard(relevant[:15], retrieved[:15])
    j_20 = calculate_jaccard(relevant[:20], retrieved[:20])
    return {'jaccard_5': j_5, 'jaccard_10': j_10, 'jaccard_15': j_15, 'jaccard_20': j_20}


def calculate_correlation(u, b):
    r_p, p_p = pearsonr(u, b)
    r_k, p_k = kendalltau(u, b)
    return round(r_p, 10), p_p, round(r_k, 10), p_k


def evaluate(r1, r2, model_name, embedding_source, epoch, lexicon, context_shift=False):
    s = '_shifted' if context_shift else ''
    with open(f'data/{model_name}_{r1}_{r2}_{embedding_source}-{epoch}_word_weights.pkl', 'rb') as doc_r:
        word_weights = pickle.load(doc_r)
    with open(f'data//yelp_reviews_lemmas_valence{s}_{lexicon}_{r1}_{r2}.pkl', 'rb') as doc_r:
        word_valences = pickle.load(doc_r)
    stars = pd.read_csv(f'data/yelp_reviews_stars_{r1}_{r2}.csv', index_col=[0]).to_dict(orient='index')
    results = dict()
    print(f'Evaluating {model_name} model on {lexicon} lexicon trained with reviews in range {r1}-{r2} with '
          f'{embedding_source} embedding vectors ...')
    for _, key in zip(tqdm(list(range(len(list(word_weights.keys()))))), word_weights.keys()):
        if key not in word_valences.keys():
            continue
        weights = sorted([(k, word_weights[key][k]) for k in word_weights[key]], key=lambda x: x[1], reverse=True)
        label = 'pos' if stars[key]['0'] > 3 else 'neg'
        valences = sorted(word_valences[key], key=lambda x: x[1], reverse=True) if label == 'pos' else \
            sorted(word_valences[key], key=lambda x: x[1])
        res = calculate_metrics([v[0] for v in valences], [w[0] for w in weights])
        res['pearson'], _, res['kendalltau'], _ = calculate_correlation([w[1] for w in sorted(weights, key=lambda x:
        x[0])], [v[1] for v in sorted([v for v in valences if v[0] in [w[0] for w in weights]], key=lambda x: x[0])])
        results[key] = res
    pd.DataFrame().from_dict(results, orient='index').to_csv(
        f'data/{model_name}_{r1}_{r2}_{embedding_source}-{epoch}_{lexicon}{s}.csv')


if __name__ == '__main__':
    evaluate(50, 500, 'SentDetect', 'w', 50, 'nrc')
    evaluate(50, 500, 'SentDetect', 'w', 50, 'yelp')
    evaluate(50, 500, 'SentDetect', 't', 50, 'nrc')
    evaluate(50, 500, 'SentDetect', 't', 50, 'yelp')
    evaluate(50, 500, 'StarDetect', 'w', 50, 'nrc')
    evaluate(50, 500, 'StarDetect', 'w', 50, 'yelp')
    evaluate(50, 500, 'StarDetect', 't', 50, 'nrc')
    evaluate(50, 500, 'StarDetect', 't', 50, 'yelp')
    evaluate(10, 500, 'SentDetect', 'w', 50, 'nrc')
    evaluate(10, 500, 'SentDetect', 'w', 50, 'yelp')
    evaluate(10, 500, 'SentDetect', 't', 50, 'nrc')
    evaluate(10, 500, 'SentDetect', 't', 50, 'yelp')
    evaluate(10, 500, 'StarDetect', 'w', 50, 'nrc')
    evaluate(10, 500, 'StarDetect', 'w', 50, 'yelp')
    evaluate(10, 500, 'StarDetect', 't', 50, 'nrc')
    evaluate(10, 500, 'StarDetect', 't', 50, 'yelp')
    # evaluate(50, 500, 'SentDetect', 'w', 50, 'nrc', True)
    # evaluate(50, 500, 'SentDetect', 'w', 50, 'yelp', True)
    # evaluate(50, 500, 'SentDetect', 't', 50, 'nrc', True)
    # evaluate(50, 500, 'SentDetect', 't', 50, 'yelp', True)
    # evaluate(50, 500, 'StarDetect', 'w', 50, 'nrc', True)
    # evaluate(50, 500, 'StarDetect', 'w', 50, 'yelp', True)
    # evaluate(50, 500, 'StarDetect', 't', 50, 'nrc', True)
    # evaluate(50, 500, 'StarDetect', 't', 50, 'yelp', True)
    # evaluate(10, 500, 'SentDetect', 'w', 50, 'nrc', True)
    # evaluate(10, 500, 'SentDetect', 'w', 50, 'yelp', True)
    # evaluate(10, 500, 'SentDetect', 't', 50, 'nrc', True)
    # evaluate(10, 500, 'SentDetect', 't', 50, 'yelp', True)
    # evaluate(10, 500, 'StarDetect', 'w', 50, 'nrc', True)
    # evaluate(10, 500, 'StarDetect', 'w', 50, 'yelp', True)
    # evaluate(10, 500, 'StarDetect', 't', 50, 'nrc', True)
    # evaluate(10, 500, 'StarDetect', 't', 50, 'yelp', True)
