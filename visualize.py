import seaborn as sns
import matplotlib.pyplot as plt
import json
import _pickle as pickle
import pandas as pd
import numpy as np
from scipy import stats


def plot_distribution(data, title, x_axis_name, x_axis_data, y_axis_name, y_axis_data):
    """
    Plots given data.
    :param data: data to be plotted
    :type data: pandas.DataFrame
    :param title: title of the plot
    :type title: str
    :param x_axis_name: name of the x axis
    :type x_axis_name: str
    :param x_axis_data: name of the data to be plotted along x axis
    :type x_axis_data: str
    :param y_axis_name: name of the y axis
    :type y_axis_name: str
    :param y_axis_data: name of the data to be plotted along y axis
    :type y_axis_data: str
    """
    sns.set(style='darkgrid', context='poster', font='Verdana')
    f, ax = plt.subplots()
    ax.set(xscale='log', yscale='log')
    plt.xlim(1e-1, 1e4)
    plt.ylim(1e-1, 1e5)
    ax.grid(False)
    sns.regplot(x_axis_data, y_axis_data, data, ax=ax, scatter_kws={'s': 250}, fit_reg=False, color='xkcd:turquoise')
    ax.set_ylabel(y_axis_name)
    ax.set_xlabel(x_axis_name)
    ax.grid(False)
    plt.title(title)
    plt.show()


def print_word_frequencies(frequencies_file):
    """
    Print ten most and least common words.
    :param frequencies_file: name of the file containing word frequencies
    :type frequencies_file: str
    """
    with open(frequencies_file, 'rb') as f:
        frequencies = pickle.load(f)
    print('=== Most common words ===')
    for word in frequencies[:10]:
        print(f'\'{word[0]}\': {str(word[1])}')
    print()
    print('=== Least common words ===')
    for word in frequencies[len(frequencies) - 10:]:
        print(f'\'{word[0]}\': {str(word[1])}')
    print()


def print_review_statistic(file_name):
    s = [0, 0, 0, 0, 0]
    with open(file_name, 'r', encoding='utf-8') as doc_r:
        line = doc_r.readline()
        i = 0
        while line != '':
            i += 1
            if i % 100000 == 0:
                print(str(i))
            review = json.loads(line)
            stars = review['stars']
            s[int(stars) - 1] += 1
            line = doc_r.readline()
    for i in range(len(s)):
        print(str(int(i) + 1) + ' stars: ' + str(s[int(i)]))
    print('1 and 2 stars: ' + str(s[0] + s[1]))
    print('1, 2 and 3 stars: ' + str(s[0] + s[1] + s[2]))
    print('4 and 5 stars: ' + str(s[3] + s[4]))
    print('3, 4 and 5 stars: ' + str(s[2] + s[3] + s[4]))


def print_review_length_statistic(lengths_file, lengths_filtered_file):
    """
    Prints caption statistics (minimum, maximum, average, median, mode).
    :param lengths_file: name of the file containing review length information
    :type lengths_file: str
    :param lengths_filtered_file: name of the file containing review length information
    :type lengths_filtered_file: str
    """
    lengths = pd.read_csv(lengths_file, sep=',', index_col=0).get_values().flatten()
    lengths_filtered = pd.read_csv(lengths_filtered_file, sep=',', index_col=0).get_values().flatten()
    print('===== Review length =====')
    print(f'Min: {round(np.min(lengths), 2)}')
    print(f'Max: {round(np.max(lengths), 2)}')
    print(f'Avg: {round(np.average(lengths), 2)}')
    print(f'Median: {round(np.median(lengths), 2)}')
    print(f'Mode: {round(stats.mode(lengths)[0][0], 2)}')
    print()
    print('===== Review length filtered =====')
    print(f'Min: {round(np.min(lengths_filtered), 2)}')
    print(f'Max: {round(np.max(lengths_filtered), 2)}')
    print(f'Avg: {round(np.average(lengths_filtered), 2)}')
    print(f'Median: {round(np.median(lengths_filtered), 2)}')
    print(f'Mode: {round(stats.mode(lengths)[0][0], 2)}')
    print()


def plot_user_review_frequencies():
    frequencies = pd.read_csv('data/user_review_frequencies.csv', sep=',', index_col=0)['0'].get_values()
    distribution_array = frequencies
    distribution = np.zeros(np.max(distribution_array) + 1, dtype=np.float32)
    distribution_index = np.array(range(np.max(distribution_array) + 1), dtype=np.float32)
    for d in distribution_array:
        distribution[d] += 1
    data = pd.DataFrame()
    data['Number of reviews'] = distribution_index
    data['Number of users'] = distribution
    plot_distribution(data, 'Number of reviews per user', 'Number of reviews', 'Number of reviews',
                      'Number of users', 'Number of users')


def plot_word_frequencies_distribution(frequencies_file):
    """
    Plots the distribution of word frequencies.
    :param frequencies_file: name of the file containing word frequencies
    :type frequencies_file: str
    """
    with open(frequencies_file, 'rb') as f:
        frequencies = pickle.load(f)
    distribution_array = [f[1] for f in frequencies]
    distribution = np.zeros(np.max(distribution_array) + 1, dtype=np.float32)
    distribution_index = np.array(range(np.max(distribution_array) + 1), dtype=np.float32)
    for d in distribution_array:
        distribution[d] += 1
    data = pd.DataFrame()
    data['Number of words'] = distribution_index
    data['Frequency'] = distribution
    plot_distribution(data, 'Word frequency distribution', 'Number of words', 'Number of words',
                      'Frequency', 'Frequency')


def plot_review_length_distribution(lengths_file):
    """
    Plots the distribution of reviews length.
    :param lengths_file: name of the file containing image captions
    :type lengths_file: str
    """
    lengths = pd.read_csv(lengths_file, sep=',', index_col=0).get_values().flatten()
    distribution_array = lengths
    distribution = np.zeros(np.max(distribution_array) + 1, dtype=np.float32)
    distribution_index = np.array(range(np.max(distribution_array) + 1), dtype=np.float32)
    for d in distribution_array:
        distribution[d] += 1
    data = pd.DataFrame()
    data['Review length'] = distribution_index
    data['Number of reviews'] = distribution
    plot_distribution(data, 'Review length distribution', 'Review length', 'Review length',
                      'Number of reviews', 'Number of reviews')


def load_values(file_name):
    return pd.read_table(file_name, sep=',')


def plot_graph_loss(values, model_name):
    data = pd.DataFrame()
    data['epoch'] = list(values['epoch'].get_values() + 1) + list(values['epoch'].get_values() + 1)
    data['loss name'] = ['training'] * len(values) + ['validation'] * len(values)
    data['loss'] = list(values['loss'].get_values()) + list(values['val_loss'].get_values())
    sns.set(style='darkgrid', context='poster', font='Verdana')
    f, ax = plt.subplots()
    sns.lineplot(x='epoch', y='loss', hue='loss name', style='loss name', dashes=False, data=data, palette='Set2')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend().texts[0].set_text('')
    plt.title(model_name)
    plt.show()


def plot_graph_f1_scores(values, model_name):
    data = pd.DataFrame()
    data['review'] = list(range(values.count()[0])) * 4
    data['cutoff'] = ['top 5'] * values.count()[0] + ['top 10'] * values.count()[0] + ['top 15'] * values.count()[0] + \
                     ['top 20'] * values.count()[0]
    data['score'] = list(values['5'].get_values()) + list(values['10'].get_values()) + list(values['15'].get_values()) \
                    + list(values['20'].get_values())
    sns.set(style='darkgrid', context='poster', font='Verdana')
    f, ax = plt.subplots()
    sns.lineplot(x='review', y='score', hue='cutoff', style='cutoff', dashes=False, data=data, palette='Set2')
    ax.set_ylabel('Score')
    ax.set_xlabel('Review')
    ax.legend().texts[0].set_text('')
    plt.title(model_name)
    plt.show()


def print_f1_scores(model_names):
    for model_name in model_names:
        print(f'===== {model_name}')
        data = load_values(f'data/{model_name}.csv')
        top_5_avg = round(np.average(data['jaccard_5'].get_values()), 5)
        top_10_avg = round(np.average(data['jaccard_10'].get_values()), 5)
        top_15_avg = round(np.average(data['jaccard_15'].get_values()), 5)
        top_20_avg = round(np.average(data['jaccard_20'].get_values()), 5)
        pearson_avg = round(np.average([d for d in data['pearson'].get_values() if not np.isnan(d)]), 5)
        kendalltau_avg = round(np.average([d for d in data['kendalltau'].get_values() if not np.isnan(d)]), 5)
        print(f'======== jaccard top 5: {top_5_avg}')
        print(f'======== jaccard top 10: {top_10_avg}')
        print(f'======== jaccard top 15: {top_15_avg}')
        print(f'======== jaccard top 20: {top_20_avg}')
        print(f'======== pearson: {pearson_avg}')
        print(f'======== kendalltau: {kendalltau_avg}')
        print()
    print()


if __name__ == '__main__':
    # print_review_statistic('data/yelp_reviews_filtered.json')
    # plot_user_review_frequencies()
    # plot_review_length_distribution('data/reviews_length.csv')
    # plot_review_length_distribution('data/reviews_length_50_500.csv')
    # plot_review_length_distribution('data/reviews_length_filtered_50_500.csv')
    # print_review_length_statistic('data/reviews_length_50_500.csv', 'data/reviews_length_filtered_50_500.csv')
    # plot_review_length_distribution('data/reviews_length_10_500.csv')
    # plot_review_length_distribution('data/reviews_length_filtered_10_500.csv')
    # print_review_length_statistic('data/reviews_length_10_500.csv', 'data/reviews_length_filtered_10_500.csv')
    # plot_word_frequencies_distribution('data/vocabulary_full_frequencies_50_500.pkl')
    # plot_word_frequencies_distribution('data/vocabulary_full_frequencies_10_500.pkl')
    # plot_graph_loss(load_values('logs/SentDetect_50_500_w.log'), 'SentDetect 50-500 w')
    # plot_graph_loss(load_values('logs/StarDetect_50_500_w.log'), 'StarDetect 50-500 w')
    # plot_graph_loss(load_values('logs/SentDetect_50_500_t.log'), 'SentDetect 50-500 t')
    # plot_graph_loss(load_values('logs/StarDetect_50_500_t.log'), 'StarDetect 50-500 t')
    # plot_graph_loss(load_values('logs/SentDetect_10_500_w.log'), 'SentDetect 10-500 w')
    # plot_graph_loss(load_values('logs/StarDetect_10_500_w.log'), 'StarDetect 10-500 w')
    # plot_graph_loss(load_values('logs/SentDetect_10_500_t.log'), 'SentDetect 10-500 t')
    # plot_graph_loss(load_values('logs/StarDetect_10_500_t.log'), 'StarDetect 10-500 t')
    # plot_graph_f1_scores(load_values('data/SentDetect_50_500_w-50_nrc.csv'), 'SentDetect 50-500 w NRC')
    # plot_graph_f1_scores(load_values('data/SentDetect_50_500_w-50_yelp.csv'), 'SentDetect 50-500 w Yelp')
    print_f1_scores(['SentDetect_50_500_w-50_nrc', 'SentDetect_50_500_w-50_yelp',
                     'SentDetect_50_500_t-50_nrc', 'SentDetect_50_500_t-50_yelp',
                     'StarDetect_50_500_w-50_nrc', 'StarDetect_50_500_w-50_yelp',
                     'StarDetect_50_500_t-50_nrc', 'StarDetect_50_500_t-50_yelp',
                     'SentDetect_10_500_w-50_nrc', 'SentDetect_10_500_w-50_yelp',
                     'SentDetect_10_500_t-50_nrc', 'SentDetect_10_500_t-50_yelp',
                     'StarDetect_10_500_w-50_nrc', 'StarDetect_10_500_w-50_yelp',
                     'StarDetect_10_500_t-50_nrc', 'StarDetect_10_500_t-50_yelp'])
