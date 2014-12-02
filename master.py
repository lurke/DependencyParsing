import os
import sys
import copy
import numpy as np
import pickle
from nltk.corpus import dependency_treebank as dp
from nltk.parse.dependencygraph import DependencyGraph
from nltk.tree import Tree
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.feature_extraction import DictVectorizer

INPATH = os.getcwd() + '/dep_treebank/'
folders = [
    '02/', '03/', '04/', '05/', '06/', '07/', '08/', '09/', '10/', '11/',\
    '12/', '13/', '14/', '15/', '16/', '17/', '18/', '19/', '20/', '21/'
]
testfiles_dir = [INPATH + folder for folder in folders]
testfiles = []
for testfile_dir in testfiles_dir:
    testfiles.append([testfile_dir + file for file in os.listdir(testfile_dir)])
currently_training = False
if len(sys.argv) > 1:
    currently_training = int(sys.argv[1])

lcontext = rcontext = 2
if len(sys.argv) > 3:
    lcontext = int(sys.argv[2])
    rcontext = int(sys.argv[3])

def tree_to_graph(tree):
    '''Converts a tree structure to a graph structure. This is for the accuracy() function.

    Args: tree: the tree to convert
    Returns: a graph representing the tree. note that this graph is really only
        useable in accuracy() (the only attribute we bother setting is 'head')
    Raises: None
    '''
    # nodes are dictionaries, which are mutable. So we copy them so we can 
    # change attributes without changing the original nodes
    tree2 = tree_map(copy.copy, tree)
    # set the head attributes of each node according to our tree structure
    def set_heads(tree, parent=0):
        n = label(tree)
        n['head'] = parent
        if isinstance(tree, Tree):
            [set_heads(child, n['address']) for child in tree]
    set_heads(tree2)

    # now we need to generate our nodelist. This requires getting all the
    # elements ("labels") of our tree and putting them in a flat list
    def all_elems(tree):
        elems = [label(tree)]
        if isinstance(tree, Tree):
            for t in tree:
                elems += all_elems(t)
        return elems

    dg = DependencyGraph()
    dg.root = dg.nodelist[0]
    all = all_elems(tree2)
    # nodelist should be ordered by address
    all.sort(key=lambda t: label(t)['address'])
    dg.nodelist += all

    return dg

def accuracy(truelist, predictedlist):
    '''Calculate the accuracy of predicted dependency trees predictedlist compared to true trees truelist

    Args:
        truelist: list of true dependency graphs
        predictedlist: list of predicted dependency graphs
    Returns: (dependency_accuracy, root_accuracy, complete_sentence_accuracy)
    Raises: None
    '''
    correct_parents = 0
    total_parents = 0
    correct_roots = 0
    total_sents = 0
    complete_sentences = 0
    for true, predicted in zip(truelist, predictedlist):
        total_sents += 1
        # if we couldn't fully reduce, we just count that as a miss
        if not predicted:
            continue
        complete = 1
        for true_node, predicted_node in zip(true.nodelist, predicted.nodelist):
            if true_node['address'] == 0:
                continue
            total_parents += 1
            if true_node['head'] == predicted_node['head']:
                correct_parents += 1
            else:
                complete = 0
            if true_node['head'] == 0 and predicted_node['head'] == 0:
                correct_roots += 1
        if complete == 1:
            complete_sentences += 1
    dep_acc = 1.*correct_parents/total_parents
    root_acc = 1.*correct_roots/total_sents
    comp_acc = 1.*complete_sentences/total_sents
    print 'total_sents', total_sents
    return (dep_acc, root_acc, comp_acc)

def tree_map(f, t):
    '''map function for Tree structures

    Args:
        f: function to apply to each element in tree
        t: tree to map
    Returns: new tree where every element e\in t is now f(e)
    Raises: None
    '''
    if isinstance(t, Tree):
        return Tree(f(t.label()), map(lambda t2: tree_map(f, t2), t))
    return f(t)

def wordify(tree):
    '''Turns a tree of nodes into a tree of words (for more useful outputs)

    Args: tree: tree to wordify
    Returns: a new tree where elements are now just words, not nodes
    Raises: None
    '''
    return tree_map(lambda w: w['word'], tree)

def label(t):
    '''Gets label of a tree (this is the value of a given point in the tree,
    or the node itself if t is a node not a tree

    Args: t: node or tree to consider
    Returns: label/node of the tree
    Raises: None
    '''
    return t.label() if isinstance(t, Tree) else t

def address(n):
    '''Gets address of node/tree n

    Args: n: node/tree to consider
    Returns: address (e.g. location in the sentence)
    Raises: None
    '''
    return label(n)['address']

def make_parent(t1, t2):
    '''Makes t1 a parent of t2

    Args: 
        t1: node/tree to make parent
        t2: node/tree to make child
    Returns: tree where t1 is parent of t2
    Raises: None
    '''
    if isinstance(t1, Tree):
        # add to list of children
        t1.append(t2)
        # unclear if necessary to sort
        t1.sort(key=lambda t: label(t)['address'])
        return t1
    else:
        return Tree(t1, [t2])

Left = 1
Right = 2
Shift = 3
class Parser(object):
    def __init__(self, model, lcontext=2, rcontext=2):
        '''init function
        
        Args:
            model: modeling class to use (either a training model or predictor)
            lcontext: how far to the left of current location to consider
            rcontext: how far to the right to consider
        Returns: Raises: None
        '''
        self.model = model
        self.lcontext = lcontext
        self.rcontext = rcontext
    
    def construction(self, T, i, y):
        '''Performs y (Left, Right, Shift) at ith position in T

        Args:
            T: array to consider
            i: location in T to look at (we con
            y: action to perform
        Returns: the next index in T to look at.
        Raises: None
        '''
        if y == Left:
            t2 = make_parent(T[i], T[i+1])
        elif y == Right:
            t2 = make_parent(T[i+1], T[i])
        elif y == Shift:
            return i+1
        T[i:i+2] = [t2]
        # should we return i or i+1 here? i means we keep looking at the 
        # new tree we just constructed and whatever's next, i+1 would mean
        # we skip this new tree now and look at the next 2 tokens in our tree.
        # both seem like they should work? (and give similar results) but
        # it's unclear if there's a meaningful difference
        return i+1

    def get_poslex(self, node, parent_addr, rel):
        '''Gets features for a single node (and determines if node is parent
        or child so as to mark the features appropriately)

        Args:
            node: what node to consider (this should be a graph node)
            parent_addr: == parent_node['address'] 
              (parent's location in sentence)
            rel: relative index of this node
        Returns: dictionary which maps the features of this node to 1's
        Raises: None
        '''
        types = ['pos', 'lex']
        # left child
        if node['address'] < parent_addr:
            types = ['ch-L-' + s for s in types]
        elif node['address'] > parent_addr:
            types = ['ch-R-' + s for s in types]
        # we have 2 attributes per node: its POS (ctag), and what word it is
        return {repr((rel, types[0], node['ctag'])): 1, repr((rel, types[1], node['word'])): 1}

    def get_single_features(self, T, i, j):
        '''Gets features for a single element of T (includes children features)

        Args: 
            T: array we're looking at
            i: index of element in T we're looking at
            j: index of the base element we're considering. This is "i" in the 
            paper (just used to calculate our relative index)
        Returns: dictionary mapping features of this element to 1's
        Raises: None
        '''
        t = T[i]
        rel = i-j
        our_addr = label(t)['address']
        dict = self.get_poslex(label(t), our_addr, rel)
        if isinstance(t, Tree):
            # look at immediate children too
            poslexes = [self.get_poslex(label(n), our_addr, rel) for n in t]
            map(dict.update, poslexes)
        return dict

    def get_contextual_features(self, T, i):
        '''Gets all the features for looking at element i of T (including context)
        
        Args:
            T: array we're considering
            i: where in T to look
        Returns: dictionary mapping features of this element and contextual
            elements to 1's (used to make our SVM matrix)
        Raises: None
        '''
        low, high = i - self.lcontext, i + self.rcontext + 1
        if low < 0:
            low = 0
        if high >= len(T):
            high = len(T)-1
        d = {}
        # the elements we're considering
        elems = range(low, high+1)
        # we add the features for each element we look at to our big dictionary
        map(lambda ind: d.update(self.get_single_features(T, ind, i)), elems)
        return d

    def parse(self, T):
        '''Parses a sentence into a dependency graph

        Args: 
            T: the list of words in the sentence (each word is in
            the form of a graph node). This list should come from 
            dp.parsed_sents()
        Returns: list of final reduced dependency trees. Generally this will 
            only contain 1 element, a fully reduced tree. However in some cases
            it may not be possible to fully reduce the tree, in which case
            the list contains the subtrees that were produced before progress
            could no longer be made
        Raises: None
        '''
        # pretty much straight from the paper
        i = 0
        no_construction = True
        while len(T) >= 1:
            if i >= len(T)-1:
                if no_construction:
                    break
                no_construction = True
                i = 0
            else:
                x = self.get_contextual_features(T, i)
                y = self.model.estimate_action(T, i, x)
                i = self.construction(T, i, y)
                if y in [Left, Right]:
                    no_construction = False
        return T

class Train:
    '''Class model used for training'''
    def __init__(self):
        '''init function

        Args: Returns: Raises: None
        '''
        self.feature_lists = {}
        self.action_lists = {}

    def estimate_action(self, T, i, x):
        '''calculates the action to perform between the ith and i+1th elems of T (using actual dependeny relations)

        Args:
            T: array of elements (as used in the paper)
            i: where in T to consider (we look at the ith and i+1th elements)
            x: feature dictionary to use for prediction
        Returns: Action to perform (Shift, Left, or Right)
        Raises: None
        '''
        node1 = T[i]
        node1children = []
        if isinstance(node1, Tree):
            # mapping through a tree iterates through its children only
            # so this will give us a list of children's addresses
            node1children = map(address, node1)
            node1 = node1.label()
        node2 = T[i+1]
        node2children = []
        if isinstance(node2, Tree):
            node2children = map(address, node2)
            node2 = node2.label()

        ret = Shift
        # in order to do a Left or Right, we require that one node is child 
        # of the other AND the node we're making child has already had all of
        # its children added to it. otherwise we Shift
        if node1['head'] == node2['address'] \
                and set(node1['deps']) == set(node1children):
            ret = Right
        if node2['head'] == node1['address'] \
                and set(node2['deps']) == set(node2children):
            ret = Left

        pos_tag = node1['ctag']
        if pos_tag not in self.feature_lists:
            self.feature_lists[pos_tag] = []
            self.action_lists[pos_tag] = []
        self.feature_lists[pos_tag].append(x)
        self.action_lists[pos_tag].append(ret)
        return ret

class Predict:
    '''Predictor model. Uses SVM(s) to predict actions'''
    def __init__(self, models):
        '''init function

        Args:
            models: return value of gen_svc() (dict from POS tag to a tuple
            of a dictvectorizer and an SVM). What this predictor will use
            for its prediction
        Returns: Raises: None
        '''
        self.models = models

    def estimate_action(self, T, i, x):
        '''estimates the action to perform between the ith and i+1th elems of T using SVMs
        
        Args:
            T: array of elements (as used in the paper)
            i: where in T to consider (we look at the ith and i+1th elements)
            x: feature dictionary to use for prediction
        Returns: Action to perform (Shift, Left, or Right)
        Raises: None
        '''
        # convert to a vector and then use our SVM predictor
        pos_tag = label(T[i])['ctag']
        dictvec = self.models[pos_tag][0]
        trained_svc = self.models[pos_tag][1]
        xmat = dictvec.transform(x)
        pred = trained_svc.predict(xmat)
        return pred[0]

class AlwaysPredict:
    '''Simple SVM emulator which always predicts the same value. A hack for the SVM library code.'''
    def __init__(self, pred):
        '''init function

        Args: pred: value this class will always predict
        Return: None
        Raises: None
        '''
        self.pred = pred

    def predict(self, x):
        '''Predicts what action to use
        
        Args: x: ignored (feature matrix)
        Return: Always the same prediction (as specified in __init__)
        Raises: None
        '''
        return self.pred

def gen_svc(train_model):
    '''Given a training model, generates the SVM (and DictVectorizer) for it

    Args: 
        train_model: a training model object. should have 2 attributes:
        feature_lists, a map from POS tag to a dictionary of features
        (the ones used in the ith decision), and action_lists, a map from
        POS tag to the action (Shift, Left, Right) chosen for the ith decision
    Returns: dictionary mapping POS tag to a vectorizer, SVM tuple
    Raises: None
    '''
    models = {}
    for pos_tag in train_model.feature_lists:
        vec = DictVectorizer()
        feature_mat = vec.fit_transform(train_model.feature_lists[pos_tag])
        trained_svc = OneVsOneClassifier(LinearSVC())
        try:
            trained_svc.fit(feature_mat, np.array(train_model.action_lists[pos_tag]))
        except ValueError:
            # occasionally we get the same action for everything with a
            # particular POS, which raises an error. so in that case we just
            # use a custom class that always predicts the same action
            trained_svc = AlwaysPredict(train_model.feature_lists[pos_tag][0])
        models[pos_tag] = (vec, trained_svc)
    return models

def do_parse(parser, sents):
    '''Given a Parser object and a list of sentences, parses each sentence
    and builds up a list of the resulting dependency trees (1 per sentence)

    Args:
        parser: Parser object to use
        sents: list of sentences (in the form of dependency graphs) to parse
    Returns:
        List of parsed trees/graphs for each sentence.
    Raises: None
    '''
    trees = []
    i = 1
    for sent in sents:
        ts = parser.parse(sent.nodelist[1:])
        print 'sentence %d' % i
        i+=1
        if len(ts) > 1:
            print "couldn't fully reduce..."
            # for now we just mark this as a failure and count it is as incorrect
            trees.append(None)
        else:
            trees.append(tree_to_graph(ts[0]))
    return trees

def main():
    '''main function. either trains the model or tests it on a dataset'''
    if currently_training:
        sents = sum([dp.parsed_sents(testfile) for testfile in testfiles], [])
        train = Train()
        p = Parser(train, lcontext, rcontext)
        trees = do_parse(p, sents)

        models = gen_svc(train)
        pkl = open('models.pkl','wb')
        pickle.dump(models, pkl)
        pkl.close()

    else:
        models = pickle.load(open('models.pkl','rb'))
        predict = Predict(models)
        p = Parser(predict, lcontext, rcontext)
        testfiles2_dir = INPATH + '23/'
        testfiles2 = [testfiles2_dir + file for file in os.listdir(testfiles2_dir)]
        sents = sum([dp.parsed_sents(testfile) for testfile in testfiles2], [])
        trees_predict = do_parse(p, sents)

        print 'ACCURACYS:'
        print accuracy(sents, trees_predict)

if __name__ == '__main__':
    main()
