from sklearn import svm
from nltk.corpus import dependency_treebank as dt
import numpy as np
from sklearn.feature_extraction import DictVectorizer as dv
import pickle
import os


'''import data'''
def import_data():

    #iterate through directory to get all the data
    '''for subdir, dirs, files in os.walk('./dep_treebank'):
        for row in files:
            for f in row:
                print f'''

    #example of what to do for each file
    path = "../../../../../Users/lurke/Documents/Harvard/Senior/CS187/final/dep_treebank"
    f = path + "/00/wsj_0001.mrg"
    t = dt.parsed_sents(f)[0]
    print t
    
    #TODO: represent this tree in the best format to gather the features
    #TODO: split data by directory into train and test

    


'''train'''
def train():
    #for each sentence in the dependency parses, construct the tree:
    features=[]
    answers = []
    data = []
    for sentence in data:
        #format sentence to pass into tree maker
        #iterate through sentence until tree is formed - one each iteration, will consider multiple pairs of words
        #each time we consider two words, get features, save as row in array with dictvectorize
        '''features, answers = make_tree('',sentence,True)
        features.append(featues)
        answers.append(answers)'''
        pass
        
    
    #example so that can do svm
    #order of features  -2,-1,0-,0+,1,2: pos,lex,ch-L-pos,ch-L-lex,ch-R-pos,ch-R-lex
    example_features1 = {'pos-2': ':', 'pos-1': 'NNS', 'pos-0': 'IN', 'pos0': 'NN', 'pos1': 'WP', 'pos1': ':'}
    example_features1.update( {'lex-2': '-', 'lex-1': 'sellers','lex-0': 'of','lex0': 'resort','lex1': 'who','lex2': '-'})
    example_features1.update({'chrrpos-1':'DT','chrrpos0':'JJ','chrlex-1':'the','chrlex0':'last'})
    example_features1.update({'chlpos1': 'VBD', 'chllex1':'were'})
    example_features2 = {'pos-2': 'NNS', 'pos-1': 'IN', 'pos-0': 'NN', 'pos0': 'WP', 'pos1': ':'}
    example_features2.update( {'lex-2': 'sellers', 'lex-1': 'of','lex-0': 'resort','lex0': 'who','lex1': '-'})
    example_features2.update({'chrrpos-2':'DT','chrrpos-0':'JJ','chrlex-2':'the','chrlex-0':'last'})
    example_features2.update({'chlpos0': 'VBD', 'chllex0':'were'})
    #this feature matrix has two entries
    example_features = [example_features1,example_features2]
    #each time we consider two words, get correct action (shift, right, left), put in the answer vector
    example_answers = np.array(['right','left'])

    #dictvectorize to turn strings into numerical values
    vec = dv()
    example_array = vec.fit_transform(example_features).toarray()
    '''array = vec.fit_transform(features).toarray()'''

    #TODO: a later goal: once we have the matrix, sort and split label (left, right, split) data 
    #so that we can run three different models

    #use sklearn svm to come up with a model
    #persist the model in a pickle
    clf = svm.SVC()
    clf.fit(example_array,example_answers)
    '''clf.fit(array,answers)'''
    pkl = open('svm.pkl','wb')
    pickle.dump(clf,pkl)
    pkl.close()

    return clf


'''test'''
def test(model,data):
    #TODO: figure out why pickle loading isnt working, so that we can save it 
    # instead of passing it in as a variable
    '''model = pickle.load('svm.pkl')'''
    trees = []

    #for each datum in the test data, estimate what the tree should be
    for datum in data:
        #TODO: format sentence in proper way from datum
        trees.append(make_tree(model, sentence))
    return trees

'''function for both train and test that iterates over a sentence, constructing the tree and either
train: outputting the features and answers as it goes along to construct the matrix for svm or 
test: outputting the predicted tree'''
def make_tree(model,sentence,train=False):
    i=1
    T=sentence
    no_construction = true
    features = []
    answers = []
    while(len(T) >= 1):
        if i == len(T):
            if no_construction: break
            no_construction = True
            i = 1
        else:
            x = get_contextual_features(T,i)
            if train:
                #TODO:get the action from the answer key
                #TODO:append feature rows and answer vector
                pass
            else:
                y = model.predict(x)
            construction(T,i,y)
            if y == 'left' or y == 'right':
                no_construction = False
    if train:
        return features, answers
    else: return T

'''the function that does the actual tree construction after an action has been selected for 
the current two words'''
def construction(T,i,y):
    #TODO: make the nodes at i and i+1 in T take the action recommended by y
    pass

'''evaluate: the function that will evaluate the test results'''
def evaluate(trees):
    for tree in tree:
        #TODO: do evaluation to see if the tree is correct
        #TODO:add these stats to overall stats
        pass
    #return overall stats


def main():
    import_data()
    model = train()
    #trees = test(model,test)
    #stats = evaluate(trees)



if __name__=="__main__":
    main()


