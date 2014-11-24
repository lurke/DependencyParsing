import nltk
from nltk.corpus import dependency_treebank as dp

FILE="/home/henrik/Dropbox/Dependency parsing/data/dep_treebank/02/wsj_0200.mrg"

def main():    
    '''
    Read and print sentence
    '''
    parsed = dp.parsed_sents(FILE)[0]
    tree = parsed.tree()
    print(tree.pprint())

    # The tags does not load correctly:
    # tagged = dp.tagged_sents(FILE)[0]
    
if __name__ == '__main__':
    main()
