from string import punctuation
from nltk import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import gensim
# from sklearn.datasets import fetch_20newsgroups
import re
import jieba


# newsgroups = fetch_20newsgroups()
eng_stopwords = set(stopwords.words('english'))

tokenizer = RegexpTokenizer(r'\s+', gaps=True)
stemmer = PorterStemmer()
translate_tab = {ord(p): u" " for p in punctuation}


def text2tokens(raw_text):
    """Split the raw_text string into a list of stemmed tokens."""
    clean_text = raw_text.lower().translate(translate_tab)
    tokens = [token.strip() for token in tokenizer.tokenize(clean_text)]
    tokens = [token for token in tokens if token not in eng_stopwords]
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return [token for token in stemmed_tokens if len(token) > 2]  # skip short tokens


def plot_difference_matplotlib(mdiff, title="", annotation=None):
    """Helper function to plot difference between models.

    Uses matplotlib as the backend."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 14))
    data = ax.imshow(mdiff, cmap='RdBu_r', origin='lower')
    plt.title(title)
    plt.colorbar(data)
    plt.show()


# dataset = [text2tokens(txt) for txt in newsgroups['data']]  # convert a documents to list of tokens

from gensim.corpora import Dictionary
# dictionary = Dictionary(documents=dataset, prune_at=None)
# dictionary.filter_extremes(no_below=5, no_above=0.3, keep_n=None)  # use Dictionary to remove un-relevant tokens
# dictionary.compactify()

# d2b_dataset = [dictionary.doc2bow(doc) for doc in dataset]  # convert list of tokens to bag of word representation


from gensim.models import LdaMulticore
num_topics = 41

# lda_fst = LdaMulticore(
#     corpus=d2b_dataset, num_topics=num_topics, id2word=dictionary,
#     workers=4, eval_every=None, passes=10, batch=True,
# )
#
# lda_snd = LdaMulticore(
#     corpus=d2b_dataset, num_topics=num_topics, id2word=dictionary,
#     workers=4, eval_every=None, passes=20, batch=True,
# )


def lda_model_values(num_topics, corpus, dictionary, text):
    x = []  # x-axis
    perplexity_values = []  # perplexity
    coherence_values = []  # coherence
    model_list = []  # Stores lda models corresponding to the number of topics to facilitate the generation of visualization pages.

    for topic in range(num_topics):
        print("topic number：", topic + 1)
        lda_model = gensim.models.LdaModel(corpus=corpus, num_topics=topic + 1, id2word=dictionary, chunksize=2000,
                                           passes=20, iterations=400)
        model_list.append(lda_model)


    return model_list  # , x, perplexity_values, coherence_values


if __name__ == "__main__":

   
    path_1 = 'C:/Users/LDA'

   
    with open(path_1 + '/cn_stopwords.txt', encoding='utf-8') as f:
        stop_words_list = f.read().split('\n')


   
    path_2 = 'C:/Users/LDA'

    f = open(path_2 + '/English.txt', 'r', encoding='utf-8')
    sentence = f.read()
    f.close()


 
    text = []

    #
    for word in jieba.lcut(sentence):
        if word not in stop_words_list:
            text.append(word)
    # print(text)

    #
    dictionary = gensim.corpora.Dictionary([text])
    corpus = [dictionary.doc2bow(tmp) for tmp in [text]]

   
    num_topics = 41
 
    model_list = lda_model_values(num_topics, corpus, dictionary, sentence)

for i in range(len(model_list)):
    print(model_list[i].print_topics(10,i+1))

mdiff, annotation = model_list[-1].diff(model_list[-1], distance='jaccard', num_words=50)

plot_difference_matplotlib(mdiff, title="Topic difference (two models)[jaccard distance]", annotation=annotation)
