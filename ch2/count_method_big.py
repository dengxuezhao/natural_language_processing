import sys
sys.path.append('..')
import numpy as np 
from common.util import preprocess, create_co_matrix, cos_similarity, ppmi, most_similar
from dataset import ptb
import time

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('counting co-occurrence')
C = create_co_matrix(corpus, vocab_size, window_size=window_size)
print('PPMI')
W = ppmi(C, verbose=True)

print('calculating SVD by sklearn')
start_time = time.time()
from sklearn.utils.extmath import randomized_svd
U,S,V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
print('elapsed time: {:.2f} sec'.format(time.time() - start_time))

word_vecs = U[:, :wordvec_size]
querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    print(query)
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)