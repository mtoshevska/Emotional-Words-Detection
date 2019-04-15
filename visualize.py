import seaborn as sns
import matplotlib.pyplot as plt
import json
import _pickle as pickle
import pandas as pd
import numpy as np


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
    Plots the distribution of captions length.
    :param lengths_file: name of the file containing image captions
    :type lengths_file: str
    """
    lengths = pd.read_csv(lengths_file, sep=',', index_col=0).get_values()
    distribution_array = lengths
    distribution = np.zeros(np.max(distribution_array) + 1, dtype=np.float32)
    distribution_index = np.array(range(np.max(distribution_array) + 1), dtype=np.float32)
    for d in distribution_array:
        distribution[d] += 1
    data = pd.DataFrame()
    data['Number of reviews'] = distribution_index
    data['Review length'] = distribution
    plot_distribution(data, 'Review length distribution', 'Number of reviews', 'Number of reviews',
                      'Review length', 'Review length')


if __name__ == '__main__':
    # print_review_statistic('data/yelp_reviews_filtered.json')
    # plot_user_review_frequencies()
    # plot_review_length_distribution('data/reviews_length.csv')
    plot_word_frequencies_distribution('data/vocabulary_full_frequencies_50_500.pkl')
    plot_word_frequencies_distribution('data/vocabulary_full_frequencies_10_500.pkl')