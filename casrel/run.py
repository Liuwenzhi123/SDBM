import os
import re
import math
import zipfile
import numpy as np
from collections import Counter, defaultdict
from itertools import permutations, chain
from gensim.models import Word2Vec
from utils import *


def build_model():
    inp_sent = Input(shape=(max_len,), dtype='int32')
    inp_ent = Input(shape=(max_len,), dtype='int32')
    inp_f_ent = Input(shape=(max_len,), dtype='float32')
    inp_t_ent = Input(shape=(max_len,), dtype='float32')
    inp_ent_dist = Input(shape=(1,), dtype='float32')
    f_ent = Lambda(lambda x: K.expand_dims(x))(inp_f_ent)
    t_ent = Lambda(lambda x: K.expand_dims(x))(inp_t_ent)

    ent_embed = Embedding(num_ent_classes, ent_emb_size)(inp_ent)
    sent_embed = Embedding(vocab_size, emb_size, weights=[w2v_embeddings], trainable=False)(inp_sent)

    x = Concatenate()([sent_embed, ent_embed])
    x = Conv1D(64, 1, padding='same', activation='relu')(x)

    f_res = layers.multiply([f_ent, x])
    t_res = layers.multiply([t_ent, x])

    conv = Conv1D(64, 3, padding='same', activation='relu')
    f_x = conv(x)
    t_x = conv(x)
    f_x = layers.add([f_x, f_res])
    t_x = layers.add([t_x, t_res])

    f_res = layers.multiply([f_ent, f_x])
    t_res = layers.multiply([t_ent, t_x])
    conv = Conv1D(64, 3, padding='same', activation='relu')
    f_x = conv(x)
    t_x = conv(x)
    f_x = layers.add([f_x, f_res])
    t_x = layers.add([t_x, t_res])

    f_res = layers.multiply([f_ent, f_x])
    t_res = layers.multiply([t_ent, t_x])
    conv = Conv1D(64, 3, padding='same', activation='relu')
    f_x = conv(x)
    t_x = conv(x)
    f_x = layers.add([f_x, f_res])
    t_x = layers.add([t_x, t_res])

    conv = Conv1D(64, 3, activation='relu')
    f_x = MaxPool1D(3)(conv(f_x))
    t_x = MaxPool1D(3)(conv(t_x))

    conv = Conv1D(64, 3, activation='relu')
    f_x = MaxPool1D(3)(conv(f_x))
    t_x = MaxPool1D(3)(conv(t_x))

    f_x = Flatten()(f_x)
    t_x = Flatten()(t_x)

    x = Concatenate()([f_x, t_x, inp_ent_dist])
    x = Dense(256, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model([inp_sent, inp_ent, inp_f_ent, inp_t_ent, inp_ent_dist], x)
    return model

train_data_dir = './data'
test_a_data_dir = './data'
test_b_data_dir = './data'

all_rel_types = set([tuple(re.split('[-_]' ,rel)) for rel in RELATIONS])
ent2idx = dict(zip(ENTITIES, range(1, len(ENTITIES) + 1)))

train_docs = Documents(train_data_dir)
test_docs = Documents(test_b_data_dir)

#Extract all relation in the training set
print("==========提取关系===========")
doc_ent_pair_ids = set()
#Extract entity pairs from all document relationships
for doc in train_docs:
    for rel in doc.rels:
        doc_ent_pair_id = (doc.doc_id, rel.ent1.ent_id, rel.ent2.ent_id)
        doc_ent_pair_ids.add(doc_ent_pair_id)


 #Extracting sentences from a document
print("==========extract a sentence===========")
sent_extractor = SentenceExtractor(sent_split_char='。', window_size=2, rel_types=RELATIONS,
                                   filter_no_rel_candidates_sents=True)
train_sents = sent_extractor(train_docs)
test_sents = sent_extractor(test_docs)

# Extracting relation candidate sets from sentences
max_len = 150
ent_pair_extractor = EntityPairsExtractor(all_rel_types, max_len=max_len)
train_entity_pairs = ent_pair_extractor(train_sents)
test_entity_pairs = ent_pair_extractor(test_sents)

# Character-level word vectors are trained using the sentence in which the candidate relation is located
word2idx = {'<pad>': 0, '<unk>': 1}
word2idx, idx2word, w2v_embeddings = train_word_embeddings(
    entity_pairs=chain(train_entity_pairs, test_entity_pairs),
    word2idx=word2idx,
    size=100,
    iter=10
)

# Generate training and test sets
print("==========准备数据集==========")
train_data = Dataset(train_entity_pairs, doc_ent_pair_ids, word2idx=word2idx, max_len=max_len, cate2idx=ent2idx)
test_data = Dataset(test_entity_pairs, word2idx=word2idx, max_len=max_len, cate2idx=ent2idx)

# model training
num_ent_classes = len(ENTITIES) + 1
ent_emb_size = 2
emb_size = w2v_embeddings.shape[-1]
vocab_size = len(word2idx)

tr_sent, tr_ent, tr_f_ent, tr_t_ent, tr_ent_dist, tr_y = train_data[:]
K.clear_session()
model = build_model()
print("==========train===========")
model.compile('adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
model.fit(x=[tr_sent, tr_ent, tr_f_ent, tr_t_ent, tr_ent_dist],
          y=tr_y, batch_size=64, epochs=4)


#model prediction
te_sent, te_ent, te_f_ent, te_t_ent, te_ent_dist, te_y = test_data[:]
preds = model.predict(x=[te_sent, te_ent, te_f_ent, te_t_ent, te_ent_dist], verbose=1)

#Result Output
submits = generate_submission(preds, test_entity_pairs, 0.5)
submit_file = 'submit_1.zip'
output_submission(submit_file, submits, test_b_data_dir)



