import pandas as pd
from nltk.corpus import wordnet
from nltk import edit_distance


def load_lexicon(lexicon_name):
    assert lexicon_name in ['afinn', 'nrc-affect', 'nrc-hashtag', 'nrc-vad', 'warriner', 'yelp-sentiment']
    if lexicon_name == 'yelp-sentiment':
        lexicon = pd.read_table('data/yelp-sentiment.txt', sep='\t', header=None, usecols=[0, 1], index_col=[0],
                                names=['', 'Valence'])
        lexicon['Valence'] = (lexicon['Valence'] - lexicon['Valence'].min()) / \
                             (lexicon['Valence'].max() - lexicon['Valence'].min())
    elif lexicon_name == 'warriner':
        lexicon = pd.read_csv('data/Ratings_Warriner_et_al.csv', usecols=[1, 2], index_col=[0], header=0,
                              names=['', 'Valence'])
    elif lexicon_name == 'nrc-vad':
        lexicon = pd.read_table('data/NRC-VAD.txt', sep='\t', index_col=[0], usecols=[0, 1])
    elif lexicon_name == 'nrc-hashtag':
        lexicon = pd.read_table('data/NRC-HashtagSentiment.txt', sep='\t', index_col=[0], usecols=[0, 1], header=None,
                                names=['', 'Valence'])
        lexicon['Valence'] = (lexicon['Valence'] - lexicon['Valence'].min()) / \
                             (lexicon['Valence'].max() - lexicon['Valence'].min())
    elif lexicon_name == 'nrc-affect':
        lexicon = pd.read_table('data/NRC-AffectIntensity.txt', sep='\t', index_col=[0], header=None, names=['Valence'])
    else:
        lexicon = pd.read_table('data/AFINN-111.txt', sep='\t', index_col=[0], header=None, names=['Valence'])
        lexicon['Valence'] = (lexicon['Valence'] - lexicon['Valence'].min()) / \
                             (lexicon['Valence'].max() - lexicon['Valence'].min())
    return lexicon


def load_lexicon_combination(lexicon_names):
    lexicon = pd.DataFrame()
    for name in lexicon_names:
        lex = load_lexicon(name)
        lexicon = pd.concat([lexicon, lex])
    return lexicon.reset_index().drop_duplicates('index').set_index('index')


def assign_valence(tokens, lexicon_names):
    if len(lexicon_names) == 1:
        lexicon = load_lexicon(lexicon_names[0])
    else:
        lexicon = load_lexicon_combination(lexicon_names)
    values = dict()
    missing = 0
    for token in tokens:
        token = str(token)
        if token in lexicon.index:
            values[token] = lexicon.loc[token].values[0]
        else:
            synonyms = wordnet.synsets(token)
            for synonym in synonyms:
                if synonym.lemmas()[0].name() in lexicon.index:
                    values[token] = lexicon.loc[synonym.lemmas()[0].name()].values[0]
                    break
            if token not in values.keys():
                values[token] = 0.5
                missing += 1
    print(f'Missing tokens: {100 * missing / len(tokens)}%')
    return values


if __name__ == '__main__':
    assign_valence(['wrong', 'travel', 'yes', 'no'], ['yelp-sentiment'])
