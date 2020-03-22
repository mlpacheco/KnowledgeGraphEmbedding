import json
import sys
import numpy as np

WORD2IDX = "/scratch1/pachecog/DRaiL/examples/argument_mining/data/word2idx.json"
EMBEDFILE = "/scratch1/pachecog/GloVe/glove.840B.300d.txt"

def get_pretrained_embedding(embed_file, emb_dim, vocab_index):
    vocab_size = len(vocab_index)
    pretrained_embeddings = np.random.uniform(-0.0025, 0.0025, (vocab_size, emb_dim))
    with open(embed_file, 'r') as f:
        for line in f:
            splits = line.strip().split(' ')
            word = splits[0].lower()
            vector = list(map(float, splits[1:]))
            size = len(list(vector))
            if word in vocab_index:
                pretrained_embeddings[vocab_index[word]] = vector
    return pretrained_embeddings


vocab_index = json.load(open(WORD2IDX))
pretrained_embeddings = get_pretrained_embedding(EMBEDFILE, 300, vocab_index)
print(pretrained_embeddings.shape)

np.save("word_embeddings.npy", pretrained_embeddings)
