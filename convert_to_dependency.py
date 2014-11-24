import os

INPATH="penn_treebank/treebank_3/parsed/mrg/wsj"
OUTPATH="dep_treebank"
SECTIONS=["00","01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24"]

def main():    
    '''
    Convert section 02-21 and 23 of wsj (Wall Street Journal) Penn Treebank parse tree bank to dependency treebank
    '''

    for section in SECTIONS:
        inpath = INPATH + "/" + section + "/"
        outpath = OUTPATH + "/" + section + "/"
        os.system("mkdir " + outpath)
        for file in os.listdir(inpath):
            os.system("java -jar pennconverter.jar < " + inpath + file + " > " + outpath + file)

if __name__ == '__main__':
    main()
