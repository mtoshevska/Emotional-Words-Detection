import os
import nltk
from nltk.parse.stanford import StanfordDependencyParser


os.environ['JAVAHOME'] = 'C:/Program Files/Java/jdk-12.0.1/bin'
nltk.internals.config_java('C:/Program Files/Java/jdk-12.0.1/bin/java')
path_to_jar = 'D:/StanfordParser/stanford-parser.jar'
path_to_models_jar = 'D:/StanfordParser/stanford-english-corenlp-2018-10-05-models.jar'


def dependency_parse(sentence):
    dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
    dependencies = dependency_parser.raw_parse(sentence).__next__()
    rel = list()
    for dependency in list(dependencies.triples()):
        rel.append([dependency[0][0].lower(), dependency[2][0].lower()])
    return rel


def relation_with_positive_intensifier(word, relations):
    POSITIVE_INTENSIFIERS = ['absolutely', 'completely', 'extremely', 'highly', 'really', 'so', 'too', 'totally',
                             'utterly', 'very', 'much', 'lots', 'pretty', 'high', 'huge', 'most', 'more', 'deeply',
                             'clearly', 'strongly']
    for rel in relations:
        if rel[0] == word.lower() and rel[1] in POSITIVE_INTENSIFIERS:
            return True
        if rel[1] == word.lower() and rel[0] in POSITIVE_INTENSIFIERS:
            return True
    return False


def relation_with_negative_intensifier(word, relations):
    NEGATIVE_INTENSIFIERS = ['scarcely', 'little', 'few', 'some', 'small', 'hardly', 'barely']
    for rel in relations:
        if rel[0] == word.lower() and rel[1] in NEGATIVE_INTENSIFIERS:
            return True
        if rel[1] == word.lower() and rel[0] in NEGATIVE_INTENSIFIERS:
            return True
    return False


def relation_with_negation(word, relations):
    NEGATIONS = ['not', 'never', 'none', 'nobody', 'nowhere', 'nothing', 'neither', 'no', 'noone', 'n\'t']
    for rel in relations:
        if rel[0] == word.lower() and rel[1] in NEGATIONS:
            return True
        if rel[1] == word.lower() and rel[0] in NEGATIONS:
            return True
    return False


def relation_with_connector(word, relations):
    CONNECTORS = ['although', 'however', 'but', 'notwithstanding', 'nevertheless', 'nonetheless', 'yet', 'instead',
                  'moreover', 'still', 'unfortunately', 'originally', 'surprisingly', 'ideally', 'apparently', 'though',
                  'despite', 'conversely', 'while', 'whereas', 'unlike']
    for rel in relations:
        if rel[0] == word.lower() and rel[1] in CONNECTORS:
            return True
        if rel[1] == word.lower() and rel[0] in CONNECTORS:
            return True
    return False


def shift_valence(sentence, lemmas, values):
    relations = dependency_parse(sentence)
    for i in range(len(lemmas)):
        if relation_with_positive_intensifier(lemmas[i], relations):
            values[i] *= 1.5
        if relation_with_negative_intensifier(lemmas[i], relations):
            values[i] *= 0.5
        if relation_with_negation(lemmas[i], relations):
            values[i] *= -1
        if relation_with_connector(lemmas[i], relations):
            values[i] *= 0
    return values


if __name__ == '__main__':
    dependency_parse('The quick brown fox jumps over the lazy dog.')