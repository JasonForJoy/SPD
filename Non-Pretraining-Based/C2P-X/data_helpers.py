import numpy as np
import random


def load_vocab(fname):
    '''
    vocab = {"I": 0, ...}
    '''
    vocab={}
    with open(fname, 'rt') as f:
        for i,line in enumerate(f):
            word = line.strip()
            vocab[word] = i
    return vocab

def load_char_vocab(fname):
    '''
    charVocab = {"U": 0, "!": 1, ...}
    '''
    charVocab={}
    with open(fname, 'rt') as f:
        for line in f:
            fields = line.strip().split('\t')
            char_id = int(fields[0])
            ch = fields[1]
            charVocab[ch] = char_id
    return charVocab

def to_vec(tokens, vocab, maxlen):
    '''
    length: length of the input sequence
    vec: map the token to the vocab_id, return a varied-length array [3, 6, 4, 3, ...]
    '''
    n = len(tokens)
    length = 0
    vec=[]
    for i in range(n):
        length += 1
        if tokens[i] in vocab:
            vec.append(vocab[tokens[i]])
        else:
            vec.append(vocab["_unk_"]) 
    return length, np.array(vec)

def load_dataset(fname, vocab, max_context_len, max_persona_len):

    dataset=[]
    with open(fname, 'rt') as f:
        for line in f:
            line = line.strip()
            fields = line.split('\t')

            # id
            c_id = fields[0]

            # context
            context = fields[1] + " _eos_"
            c_tokens = context.split(' ')[:max_context_len]  # select the head max_context_len tokens in every context
            c_len, c_vec = to_vec(c_tokens, vocab, max_context_len)

            # matched persona
            if fields[2] != "NA":
                personas = fields[2].split("|")
                for index, persona in enumerate(personas):
                    p_id = "1." + str(index)
                    persona = persona + " _eos_"
                    p_tokens = persona.split(' ')[:max_persona_len]  # select the head max_persona_len tokens in every persona
                    p_len, p_vec = to_vec(p_tokens, vocab, max_persona_len)
                    dataset.append((c_id, c_tokens, c_vec, c_len, 1.0, p_id, p_tokens, p_vec, p_len))
            
            # mismatched persona
            if fields[3] != "NA":
                personas = fields[3].split("|")
                for index, persona in enumerate(personas):
                    ps_id = "0." + str(index)
                    persona = persona + " _eos_"
                    p_tokens = persona.split(' ')[:max_persona_len]  # select the head max_persona_len tokens in every persona
                    p_len, p_vec = to_vec(p_tokens, vocab, max_persona_len)
                    dataset.append((c_id, c_tokens, c_vec, c_len, 0.0, p_id, p_tokens, p_vec, p_len))
   
    return dataset


def normalize_vec(vec, maxlen):
    '''
    pad the original vec to the same maxlen
    [3, 4, 7] maxlen=5 --> [3, 4, 7, 0, 0]
    '''
    if len(vec) == maxlen:
        return vec

    new_vec = np.zeros(maxlen, dtype='int32')
    for i in range(len(vec)):
        new_vec[i] = vec[i]
    return new_vec


def charVec(tokens, charVocab, maxlen, maxWordLength):
    '''
    chars = np.array( (maxlen, maxWordLength) )    0 if not found in charVocab or None
    word_lengths = np.array( maxlen )              1 if None
    '''
    n = len(tokens)
    if n > maxlen:
        n = maxlen

    chars =  np.zeros((maxlen, maxWordLength), dtype=np.int32)
    word_lengths = np.ones(maxlen, dtype=np.int32)
    for i in range(n):
        token = tokens[i][:maxWordLength]
        word_lengths[i] = len(token)
        row = chars[i]
        for idx, ch in enumerate(token):
            if ch in charVocab:
                row[idx] = charVocab[ch]

    return chars, word_lengths


def batch_iter(data, batch_size, num_epochs, max_context_len, max_persona_len,
               charVocab, max_word_length, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            random.shuffle(data)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            x_context = []
            x_context_len = []
            x_persona = []
            x_persona_len = []

            x_labels = []
            x_id_pairs = []
            
            x_context_char = []
            x_context_char_len = []         
            x_persona_char = []
            x_persona_char_len = []

            for rowIdx in range(start_index, end_index):
                c_id, c_tokens, c_vec, c_len, label, p_id, p_tokens, p_vec, p_len = data[rowIdx]

                # normalize c_vec
                new_c_vec = normalize_vec(c_vec, max_context_len)
                x_context.append(new_c_vec)
                x_context_len.append(c_len)

                # normalize p_vec
                new_p_vec = normalize_vec(p_vec, max_persona_len)
                x_persona.append(new_p_vec)
                x_persona_len.append(p_len)

                x_labels.append(label)
                x_id_pairs.append((c_id, p_id, int(label)))

                # normalize us_CharVec
                cCharVec, cCharLen = charVec(c_tokens, charVocab, max_context_len, max_word_length)
                x_context_char.append(cCharVec)
                x_context_char_len.append(cCharLen)

                # normalize ps_CharVec
                pCharVec, pCharLen = charVec(p_tokens, charVocab, max_persona_len, max_word_length)
                x_persona_char.append(pCharVec)
                x_persona_char_len.append(pCharLen)

            yield np.array(x_context), np.array(x_context_len), np.array(x_persona), np.array(x_persona_len), \
                  np.array(x_labels), x_id_pairs, \
                  np.array(x_context_char), np.array(x_context_char_len), np.array(x_persona_char), np.array(x_persona_char_len)
