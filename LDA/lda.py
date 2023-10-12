import gensim
import jieba
import re
import matplotlib.pyplot as plt
import matplotlib
from pylab import xticks,yticks,np
from gensim.topic_coherence import direct_confirmation_measure
from my_custom_module import custom_log_ratio_measure
import pyLDAvis
import pyLDAvis.gensim
#from importlib import reload
from gensim.models import CoherenceModel

direct_confirmation_measure.log_ratio_measure = custom_log_ratio_measure

#plt.switch_backend('agg')



#Model Generation Functions
def lda_model_values(num_topics, corpus, dictionary, text):
    x = [] # x-axis
    perplexity_values = [] # perplexity
    coherence_values = []   # coherence
    model_list = [] # Stores lda models corresponding to the number of topics to facilitate the generation of visualization pages.

    for topic in range(num_topics):
        print("topic number：", topic+1)
        lda_model = gensim.models.LdaModel(corpus=corpus, num_topics=topic+1, id2word = dictionary, chunksize = 2000, passes=20, iterations = 400)
        model_list.append(lda_model)
        x.append(topic+1)
        perplexity_values.append(lda_model.log_perplexity(corpus))
        coherencemodel = gensim.models.CoherenceModel(model=lda_model, texts=text, dictionary=dictionary, coherence='c_v')
        #print(coherencemodel.get_coherence())
        coherence_values.append(coherencemodel.get_coherence())
        print('The perplexity of the model is：{},The coherence is：{}'.format(perplexity_values,coherence_values))
        print("Evaluation of the theme completed\n")
        # instantiate topic coherence model


        
    return model_list, x, perplexity_values, coherence_values

if __name__ == '__main__':

    #Folders for stop words
    path_1 = 'C:/Users/LDA'

    #cn_stopwords.txtis the name of the deactivation word file, utf-8 is its encoding, the file is one word per line
    with open(path_1+'/cn_stopwords.txt',encoding='utf-8') as f:
        stop_words_list = f.read().split('\n')
    #print(stop_words)


    #Path to the data file
    path_2 = 'C:/LDA'

    f = open(path_2+'/English.txt','r',encoding='utf-8')
    sentence = f.read()
    f.close()



    text = []

    #Remove stop words from content
    for word in jieba.lcut(sentence):
        if word not in stop_words_list:
            text.append(word)
    #print(text)


    #Generate data needed for models such as bag of words
    dictionary = gensim.corpora.Dictionary([text])
    corpus = [dictionary.doc2bow(tmp) for tmp in [text]]



    """Below is the visualization of the drawing code"""
    #Starting at 1 and looping to topic 20, test the confusion and coherence of the number of topics from 1-20
    num_topics = 20
    # Calling the Model Generation Function
    model_list, x, perplexity_values, coherence_values = lda_model_values(num_topics, corpus, dictionary, sentence) 

    
    #Output the distribution of topics for the best model, model_list[15] is the 16th model generated
    for i in range(len(model_list)):
        print(model_list[i].print_topics(10,i+1))
    #ii defaults to the largest number of the last loop above.

    # Plotting Confusion and Consistency Line Plots
    fig = plt.figure(figsize=(15,5))
    plt.rcParams['font.sans-serif']=['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(x, perplexity_values, marker="o")
    plt.title("")
    plt.xlabel('')
    plt.ylabel('')
    xticks(np.linspace(1, num_topics, num_topics, endpoint=True)) # Ensure that the x-axis scale is 1



    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(x, coherence_values, marker="o")
    plt.title("")
    plt.xlabel("")
    plt.ylabel("")
    xticks(np.linspace(1, num_topics, num_topics, endpoint=True))


    plt.show()

    vis_data = pyLDAvis.gensim.prepare(model_list[i], corpus=corpus, dictionary=dictionary, mds='mmds')
    pyLDAvis.show(vis_data)


