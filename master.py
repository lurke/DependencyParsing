import os
import numpy as np
import pickle
from nltk.corpus import dependency_treebank as dp
from nltk.parse.dependencygraph import DependencyGraph
import copy
from nltk.tree import Tree
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.feature_extraction import DictVectorizer

INPATH = os.getcwd() + '/dep_treebank/'
folders = ['02/', '03/', '04/', '05/', '06/', '07/', '08/', '09/', '10/', '11/', '12/']
testfiles_dir = [INPATH + folder for folder in folders]
testfiles = []
for testfile_dir in testfiles_dir:
    testfiles.append([testfile_dir + file for file in os.listdir(testfile_dir)])
currently_training = True

def tree_to_graph(tree):
    tree2 = tree_map(copy.copy, tree)
    def set_heads(tree, parent=0):
        n = label(tree)
        n['head'] = parent
        if isinstance(tree, Tree):
            [set_heads(child, n['address']) for child in tree]
    set_heads(tree2)

    def all_elems(tree):
        elems = [label(tree)]
        if isinstance(tree, Tree):
            for t in tree:
                elems += all_elems(t)
        return elems

    dg = DependencyGraph()
    dg.root = dg.nodelist[0]
    all = all_elems(tree2)
    all.sort(key=lambda t: label(t)['address'])
    dg.nodelist += all

    return dg

def accuracy(truelist, predictedlist):
    '''
    calculate the accuracy of predicted dependency trees predictedlist compared to true trees truelist
    '''
    correct_parents = 0
    total_parents = 0
    correct_roots = 0
    total_sents = 0
    complete_sentences = 0
    for true, predicted in zip(truelist, predictedlist):
        total_sents += 1
        if not predicted:
            continue
        assert(len(true.nodelist) == len(predicted.nodelist))
        complete = 1
        for true_node, predicted_node in zip(true.nodelist, predicted.nodelist):
            assert true_node['address'] == predicted_node['address']
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
    '''map function for Tree structures'''
    if isinstance(t, Tree):
        return Tree(f(t.label()), map(lambda t2: tree_map(f, t2), t))
    return f(t)

def wordify(tree):
    return tree_map(lambda w: w['word'], tree)

def label(t):
    return t.label() if isinstance(t, Tree) else t

def address(n):
    return label(n)['address']

def make_parent(t1, t2):
    '''Makes t1 a parent of t2'''
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
    def __init__(self, model, context=2):
        self.model = model
        self.context = context
    
    def construction(self, T, i, y):
        '''Performs y (Left, Right, Shift) at ith position in T

        Returns: the next index to look at.'''
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
        or child so as to mark the features appropriately)'''
        types = ['pos', 'lex']
        # left child
        if node['address'] < parent_addr:
            types = ['ch-L-' + s for s in types]
        elif node['address'] > parent_addr:
            types = ['ch-R-' + s for s in types]
        return {repr((rel, types[0], node['ctag'])): 1, repr((rel, types[1], node['word'])): 1}

    def get_single_features(self, T, i, j):
        '''Gets features for a single element of T (includes children features)

        Args: T: array we're looking at
        i: index of element in T we're looking at
        j: index of the base element we're considering. This is "i" in the 
        paper (just used to calculate our relative index)
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
        '''Gets all the features for looking at element i of T (including context)'''
        low, high = i - self.context, i + self.context + 1
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
        '''Parses a sentence into a dependency graph'''
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

class TestModel:
    def estimate_action(self, T, i, x):
        return Left


class Train:
    def __init__(self):
        self.feature_lists = {}
        self.action_lists = {}

    def estimate_action(self, T, i, x):
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
    def __init__(self, models):
        self.models = models

    def estimate_action(self, T, i, x):
        # convert to a vector and then use our SVM predictor
        pos_tag = label(T[i])['ctag']
        dictvec = self.models[pos_tag][0]
        trained_svc = self.models[pos_tag][1]
        xmat = dictvec.transform(x)
        pred = trained_svc.predict(xmat)
        return pred[0]

def gen_svc(train_model):
    '''Given a training model, generates the SVM (and DictVectorizer) for it'''
    models = {}
    for pos_tag in train_model.feature_lists:
        vec = DictVectorizer()
        feature_mat = vec.fit_transform(train_model.feature_lists[pos_tag])
        trained_svc = OneVsOneClassifier(LinearSVC())
        trained_svc.fit(feature_mat, np.array(train_model.action_lists[pos_tag]))
        models[pos_tag] = (vec, trained_svc)
    return models

def do_parse(p, l):
    '''Given a Parser object and a list of sentences, parses each sentence
    and builds up a list of the resulting dependency trees (1 per sentence)
    '''
    trees = []
    i = 1
    for sent in l:
        ts = p.parse(sent.nodelist[1:])
        print 'sentence %d' % i
        i+=1
        if len(ts) > 1:
            print "couldn't fully reduce..."
            #trees.append(map(lambda t: str(wordify(t)), ts))
            trees.append(None)
        else:
            #print tree_map(lambda w: w['word'], ts[0])
            trees.append(tree_to_graph(ts[0]))
#            trees.append(wordify(ts[0]))
    return trees

def main():
    if currently_training:
        sents = sum([dp.parsed_sents(testfile) for testfile in testfiles], [])
        train = Train()
        p = Parser(train)
        trees = do_parse(p, sents)

        models = gen_svc(train)
        pkl = open('models.pkl','wb')
        pickle.dump(models, pkl)
        pkl.close()

    else:
        models = pickle.load(open('models.pkl','rb'))
        predict = Predict(models)
        p = Parser(predict)
        testfiles2 = [INPATH + '/23/wsj_2300.mrg']
        sents = sum([dp.parsed_sents(testfile) for testfile in testfiles2], [])
        trees_predict = do_parse(p, sents)

        print 'ACCURACYS:'
        print accuracy(sents, trees_predict)

        correct = 0
        for predict,actual in zip(trees_predict, sents):

            if predict == actual.tree():
                correct += 1
    #            print train
    #            print actual.tree()
                pass

        print correct, len(sents)


if __name__ == '__main__':
    main()
