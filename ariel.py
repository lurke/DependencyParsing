import os
import copy
import nltk
from nltk.corpus import dependency_treebank as dt
from nltk.parse import DependencyGraph
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn import multiclass

data_path = "/Users/Ariel/Dropbox/Dependency parsing/data/dep_treebank/"
folders_to_train = ["02", "03", "04", "05", "06", "07", "08", "09", 
                    "10", "11", "12", "13", "14", "15", "16", "17", 
                    "18", "19", "20", "21"]
folders_to_test = ["00", "01", "22", "24"]

LEFT = "L"
RIGHT = "R"
SHIFT = "S"

def data_file_paths_for_folders(folders):
    """
    Takes in a list of strings corresponding to folders containing data files.
    Returns a list of filepaths for data files inside said folders.
    """
    filepaths = []
    for folder in folders:
        location = data_path + folder + "/"
        filepaths.extend([location + filename for filename in os.listdir(location)])
    return filepaths

def flattened_node_list(graph):
    """
    Takes an instance of DependencyGraph corresponding to a parsed sentence.
    Flattens into a list of DependencyGraph instances, each with a different
    word from the sentence as its root node (and no children).
    """
    nodelist = copy.copy(graph.nodelist[1:])
    flattened = []
    for node in nodelist:
        node["deps"] = []
        node["head"] = 0
        node["address"] = 1
        new_graph = DependencyGraph()
        new_graph.nodelist.append(node)
        new_graph.root = node
        flattened.append(new_graph)
    return flattened

def get_contextual_features(T, i):
    """
    Takes a list of DependencyGraph instances, corresponding to the state of a
    sentence as it is being parsed into a dependency tree. Returns the contextual
    features for the target subtrees at indices i and i + 1 as a dictionary.
    """
    features = {}
    features["-0|lex"] = T[i].root["word"]
    features["-0|POS"] = T[i].root["ctag"]
    features["+0|lex"] = T[i + 1].root["word"]
    features["+0|POS"] = T[i + 1].root["ctag"]
    return features

def get_classification(T, i, graph):
    """
    Takes a list of DependencyGraph instances, corresponding to the state of a
    sentence as it is being parsed into a dependency tree, as well as the index 
    of the target nodes (i and i + 1), and the fully parsed DependencyGraph for
    the sentence. Determines correct construction action for target nodes in T
    based on their relationship in the full dependency tree.
    ========== INCOMPLETE ==========
    """
    return SHIFT

def construction(T, i, classification):
    """
    Takes a list of DependencyGraph instances, corresponding to the state of a
    sentence as it is being parsed into a dependency tree, as well as the index 
    of the target nodes (i and i + 1), and an action to take. Does the construction
    action in place on T.
    ========== INCOMPLETE ==========
    """
    pass

def train_model(folders):
    """
    Takes a list of folders from which to draw data files to train the model.
    Parses sentences in a similar way to when testing, by iteratively looking at
    target nodes in the remaining subtrees of the sentence. For each pair, the
    algorithm derives a list of features and a correct construction action. Once
    there are all found it uses them to generate a model, which is returned.
    ========== INCOMPLETE ==========
    """
    raw_features = []
    classifications = []
    for filepath in data_file_paths_for_folders(folders):
        for sentence in dt.parsed_sents(filepath):
            T = flattened_node_list(sentence)
            i = 0
            no_construction = True
            while len(T) >= 1:
                if i == len(T) - 1:
                    if no_construction:
                        break
                    no_construction = True
                    i = 0
                else:
                    target_features = get_contextual_features(T, i)
                    target_classification = get_classification(T, i, sentence)
                    raw_features.append(target_features)
                    classifications.append(target_classification)
                    construction(T, i, target_classification)
                    if target_classification != SHIFT:
                        no_construction = False
                i += 1
    vectorizer = DictVectorizer()
    feature_matrix = vectorizer.fit_transform(raw_features)
    feature_names = vectorizer.get_feature_names()
    model = multiclass.OneVsOneClassifier(svm.LinearSVC())
    model.fit(feature_matrix, classifications)
    return vectorizer, model

def estimate_classification(raw_features, vectorizer, model):
    """
    Takes in a dictionary of contextual features, a DictVectorizer, and an SVM
    model, and returns an estimated construction action for those features.
    """
    features = vectorizer.transform(raw_features)
    prediction = model.predict(features)
    return prediction[0][0]

def test_model(folders, vectorizer, model):
    """
    Takes a list of folders from which to draw data files to test the model.
    Parses sentences by iteratively looking at target nodes in the remaining 
    subtrees of the sentence. For each pair, the algorithm derives a list of 
    features, and uses the model to predict a construction action. The function
    then analyzes the performance of the model and prints out results.
    ========== INCOMPLETE ==========
    """
    for filepath in data_file_paths_for_folders(folders):
        for sentence in dt.parsed_sents(filepath):
            T = flattened_node_list(sentence)
            i = 0
            no_construction = True
            while len(T) >= 1:
                if i == len(T) - 1:
                    if no_construction:
                        break
                    no_construction = True
                    i = 0
                else:
                    target_features = get_contextual_features(T, i)
                    target_classification = estimate_classification(target_features, vectorizer, model)
                    construction(T, i, target_classification)
                    if target_classification != SHIFT:
                        no_construction = False
                i += 1

def main():
    """
    Trains a model based on some of the data files, then tests it on a different
    set of data files. Outputs performance metrics measuring how well the model
    predicted the correct sentence parses into dependency trees.
    """
    vectorizer, model = train_model(["00"])
    test_model(["00"], vectorizer, model)

if __name__ == "__main__":
    main()
