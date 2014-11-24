from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
import copy
import scipy
import pickle
import nltk
from nltk.corpus import dependency_treebank as dp
import os

INPATH = "/home/henrik/Dropbox/Dependency parsing/data/dep_treebank/"
SECTIONS=["02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21"]
SECTIONS=["02"] # only use these sections for now
testfile = INPATH + "23/wsj_2300.mrg" # only using one test file for now

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
        for true_node, predicted_node in zip(true.nodelist, predicted.nodelist):
            if true_node['address'] == 0:
                continue
            complete = 1
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
    return (dep_acc, root_acc, comp_acc)                     
        
def estimate_action(trained_model, features,  x):
    '''
    returns the predicted action according to SVM trained_model, given features x
    args:
    features - the list of features that were used in estimating the trained_model
    '''
    xvect = map(lambda i: 1 if x.has_key(i) else 0, features)
    action = trained_model.predict(xvect)
    return action[0]
        
def nodes(dp_graph):
    '''
    return list of addresses of nodes in dp_graph
    '''
    return [ dp_graph.nodelist[i]['address'] for i in range(0, len(dp_graph.nodelist)) ]
    
def erase_structure(dp_graph):
    '''
    create a copy of dp_graph with all nodes as children of the top node
    '''
    erased_dp_graph = copy.deepcopy(dp_graph)

    for node in erased_dp_graph.nodelist:
        if node['address'] == 0:
            node['deps'] = nodes(dp_graph)
        else:
            node['head'] = 0
            node['deps'] = []
    return erased_dp_graph
            

def add_arc(dp_graph, head_address, mod_address):
    """
    Adds an arc from the node specified by head_address to the
    node specified by the mod address.
    """
    for node in dp_graph.nodelist:
        if node['address'] == head_address and (mod_address not in node['deps']):
            node['deps'].append(mod_address)
        if node['address'] == mod_address:
            node['head'] = head_address

def construction(T, Z, i, y):
    '''
    update T and i with action y
    '''
    address1 = Z[i]
    address2 = Z[i+1]
    if y == 'S':
        i += 1
    if y == 'L':
        add_arc(T, address1, address2)
        Z.pop(i+1)
    if y == 'R':
        add_arc(T, address2, address1)
        Z.pop(i)
    if i > len(Z)-2:
        i = 1
    return (T, Z, i)
  
def get_contextual_features(T, Z, i, l , r):
    '''
    returns dictionary of contextual features in terms of triples (pos,type,value)

    args:
    T - dependency graph
    Z - list of top nodes
    i - index
    (l,r) - context length
    '''
    nl = T.nodelist
    contextual_features = {}
    for k in range(-l, r + 2):
        if 0 < i + k < len(Z):
            j = Z[i + k]
            if k < 0:
                position = str(k)
            elif k == 0:
                position = "0-"
            elif k == 1:
                position = "0+"
            else:
                position = str(k-1)
            contextual_features["(" + position +", pos, "+  str(nl[j]['ctag']) + ")"] = 1
            contextual_features["(" + position +", lex, "+  str(nl[j]['word']) + ")"] = 1
            for dep in nl[j]['deps']:
                if dep < j:
                    contextual_features["(" + position +", ch-L-pos, "+  str(nl[dep]['ctag']) + ")"] = 1
                    contextual_features["(" + position +", ch-L-lex, "+  str(nl[dep]['word']) + ")"] = 1
                else:
                    contextual_features["(" + position +", ch-R-pos, "+  str(nl[dep]['ctag']) + ")"] = 1
                    contextual_features["(" + position +", ch-R-lex, "+  str(nl[dep]['word']) + ")"] = 1
    return contextual_features
    
def fullsubtree(model, T, address):
    '''
    returns true if dependency graph T contains a full subtree according to model at address
    '''
    fullsubtree = True
    ml = model.nodelist
    tl = T.nodelist
    desc = [address]
    while desc != []:
        newdesc = []
        for d in desc:
            if set(ml[d]['deps']) == set(tl[d]['deps']):
                newdesc += ml[d]['deps']
            else:
                fullsubtree = False
        desc = newdesc
    return fullsubtree
        
def get_action(model, T, Z, i):
    '''
    determines whether to shift, left or right
    '''
    ml = model.nodelist
    tl = T.nodelist
    address1 = Z[i]
    address2 = Z[i+1]
    if ml[address1]['head'] == address2 and fullsubtree(model, T, address1):
        action = 'R'
    elif ml[address2]['head'] == address1 and fullsubtree(model, T, address2):
        action = 'L'
    else:
        action = 'S'
    return action

def train():
    '''
    train model
    '''
    Y = []
    X = []
    print "Main loop"
    for section in SECTIONS:
        print "section " + str(section)
        inpath = INPATH + "/" + section + "/"
        for file in os.listdir(inpath):
            print "file " + str(file)
            file = inpath + file
            parsed_file = dp.parsed_sents(file)
            nsents = 1
            for sentence in parsed_file:
                nsents += 1    
                Z = nodes(sentence)
                T = erase_structure(sentence)
                i = 1
                j = 1
                while len(Z) > 2 and j < 200:
                    j += 1
                    y = get_action(sentence, T, Z, i)
                    x = get_contextual_features(T, Z, i, 2, 2)
                    X.append(x)
                    Y.append(y)
                    (T, Z, i) = construction(T, Z, i, y)
    vec = DictVectorizer()
    print "Converting features to sparse matrix..."
    xx = vec.fit_transform(X)
    features = vec.get_feature_names()
    trained_model = LinearSVC()
    print "SVM learning..."
    trained_model.fit(xx, Y)
    print "Saving..."
    with open('train.p','w') as f:
        pickle.dump(trained_model,f)
    with open('features.p','w') as f:
        pickle.dump(features,f)
    print "Training completed"

def test():
    '''
    test model
    '''
    # load trained model and features
    with open('train.p','r') as file:
        trained_model=pickle.load(file)
    with open('features.p','r') as file:
        features=pickle.load(file)
    # load test data
    truelist = dp.parsed_sents(testfile) 
    predictedlist = []
    # main loop
    print "Running main loop..."
    k = 1
    for true in truelist:
        k += 1
        print str(k) + " of " + str(len(truelist)) + " sentences"
        # erase tree structure
        Z = nodes(true)
        T = erase_structure(true)
        # initialize
        i = 1
        X = []
        j = 1
        Y = ""
        no_construction = False
        j = 1
        # build dependency structure
        print 
        while len(Z) > 2 and j < 200: # the 200 is used to prevent infinite looping for now
            j += 1
            if i == 1:
                if no_construction == True:
                    break
                no_construction = True
            x = get_contextual_features(T, Z, i, 2, 2)
            y = estimate_action(trained_model, features, x)
            Y += y
            (T, Z, i) = construction(T, Z, i, y)
            if y != 'S':
                no_construction = False
        predictedlist.append(T)
    # measure performance
    print "Calculating performance..."
    (dep_acc, root_acc, comp_acc) = accuracy(truelist, predictedlist)
    # print output
    print "Dependency accuracy: " + str(dep_acc)
    print "Root accuracy: " + str(root_acc)
    print "Complete accuracy: " + str(comp_acc)

def main():
    train()
    test()

if __name__ == '__main__':
    main()
