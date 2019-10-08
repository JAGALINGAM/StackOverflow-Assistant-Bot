import nltk
import pickle
import re
import numpy as np
import pandas as pd

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'starspace_embedding.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embedding(embeddings_path):
   
    starspace_embeddings = {}
    data_frame=pd.read_csv("starspace_embedding.tsv",sep="\t",header=None)
    arr = np.array(data_frame)
    for i in range(arr.shape[0]):
      starspace_embeddings[arr[i][0]] = []
      for j in range(1,arr.shape[1]):
        starspace_embeddings[arr[i][0]].append(float(arr[i][j]))
      starspace_embeddings[arr[i][0]] = np.array(starspace_embeddings[arr[i][0]],dtype=float)
        
    dim = starspace_embeddings["using"].shape[0]
   
    
    return starspace_embeddings,dim
	
    # remove this when you're done
    # remove this when you're done
    raise NotImplementedError(
        "Open utils.py and fill with your code. In case of Google Colab, download"
        "(https://github.com/hse-aml/natural-language-processing/blob/master/project/utils.py), "
        "edit locally and upload using '> arrow on the left edge' -> Files -> UPLOAD")


def question_to_vec(question, embeddings, dim):
   
    res = np.zeros(dim)
    count = 0
    temp = False
    for i in question.split():
        if i not in embeddings :
            continue
        else:
            res = [x+y for x,y in zip(res,embeddings[i])]
            count+=1
            temp = True
        
    if len(question)==0 or temp==False:
      return res
    
    else:
      return [i/count for i in res]
	
    # remove this when you're done
    raise NotImplementedError(
        "Open utils.py and fill with your code. In case of Google Colab, download"
        "(https://github.com/hse-aml/natural-language-processing/blob/master/project/utils.py), "
        "edit locally and upload using '> arrow on the left edge' -> Files -> UPLOAD")


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
