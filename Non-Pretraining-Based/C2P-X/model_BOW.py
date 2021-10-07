import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS

def get_embeddings(vocab):
    print("get_embedding")
    initializer = load_word_embeddings(vocab, FLAGS.embedding_dim)
    return tf.constant(initializer, name="word_embedding")

def get_char_embedding(charVocab):
    print("get_char_embedding")
    char_size = len(charVocab)
    embeddings = np.zeros((char_size, char_size), dtype='float32')
    for i in range(1, char_size):
        embeddings[i, i] = 1.0

    return tf.constant(embeddings, name="word_char_embedding")

def load_embed_vectors(fname, dim):
    # vectors = { 'the': [0.2911, 0.3288, 0.2002,...], ... }
    vectors = {}
    for line in open(fname, 'rt'):
        items = line.strip().split(' ')
        if len(items[0]) <= 0:
            continue
        vec = [float(items[i]) for i in range(1, dim+1)]
        vectors[items[0]] = vec

    return vectors

def load_word_embeddings(vocab, dim):
    vectors = load_embed_vectors(FLAGS.embedded_vector_file, dim)
    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size, dim), dtype='float32')
    for word, code in vocab.items():
        if word in vectors:
            embeddings[code] = vectors[word]
        #else:
        #    embeddings[code] = np.random.uniform(-0.25, 0.25, dim) 

    return embeddings 


class BOW(object):
    def __init__(
      self, max_context_len, max_persona_len, num_layer, vocab_size, embedding_size, vocab, rnn_size, maxWordLength, charVocab, l2_reg_lambda=0.0):

        self.context = tf.placeholder(tf.int32, [None, max_context_len], name="context")
        self.context_len = tf.placeholder(tf.int32, [None], name="context_len")
        self.persona = tf.placeholder(tf.int32, [None, max_persona_len], name="persona")
        self.persona_len = tf.placeholder(tf.int32, [None], name="persona_len")

        self.target = tf.placeholder(tf.float32, [None], name="target")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.c_charVec = tf.placeholder(tf.int32, [None, max_context_len, maxWordLength], name="context_char")
        self.c_charLen = tf.placeholder(tf.int32, [None, max_context_len], name="context_char_len")
        self.p_charVec = tf.placeholder(tf.int32, [None, max_persona_len, maxWordLength], name="persona_char")
        self.p_charLen = tf.placeholder(tf.int32, [None, max_persona_len], name="persona_char_len")

        l2_loss = tf.constant(1.0)

        # =============================== Embedding layer ===============================
        with tf.name_scope("embedding"):
            W = get_embeddings(vocab)
            context_embedded = tf.nn.embedding_lookup(W, self.context)  # [batch_size, max_context_len, word_dim]
            persona_embedded = tf.nn.embedding_lookup(W, self.persona)  # [batch_size, max_persona_len, word_dim]
            context_embedded = tf.nn.dropout(context_embedded, keep_prob=self.dropout_keep_prob)
            persona_embedded = tf.nn.dropout(persona_embedded, keep_prob=self.dropout_keep_prob)
            print("context_embedded: {}".format(context_embedded.get_shape()))
            print("persona_embedded: {}".format(persona_embedded.get_shape()))


        # =============================== Encoding layer ===============================
        with tf.variable_scope("encoding_layer") as vs:
            mask_c = tf.sequence_mask(self.context_len, max_context_len, dtype=tf.float32)  # [batch_size, max_context_len]
            mask_c = tf.expand_dims(mask_c, -1)  # [batch_size, max_context_len, 1]
            final_context = tf.reduce_max(context_embedded * mask_c, axis=1)

            mask_p = tf.sequence_mask(self.persona_len, max_persona_len, dtype=tf.float32)  # [batch_size, max_persona_len]
            mask_p = tf.expand_dims(mask_p, -1)  # [batch_size, max_persona_len, 1]
            final_persona = tf.reduce_max(persona_embedded * mask_p, axis=1)
            print("establish BOW encoder")
            
        
        # =============================== Matching layer ===============================
        with tf.variable_scope("matching_layer") as vs:
            output_dim = final_context.get_shape()[-1].value
            A_matrix = tf.get_variable('A_matrix_v', shape=[output_dim, output_dim], initializer=tf.orthogonal_initializer(), dtype=tf.float32)

            similarity = tf.matmul(final_context, A_matrix)                  # [batch_size, dim]
            similarity = tf.reduce_sum(similarity * final_persona, axis=-1)  # [batch_size, ]
            print("shape of similarity: {}".format(similarity.get_shape()))          


        # =============================== Prediction layer ===============================
        with tf.variable_scope("prediction_layer") as vs:
            logits = similarity
            self.probs = tf.sigmoid(logits, name="prob")   # [batch_size, ]

            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.target)
            self.mean_loss = tf.reduce_mean(losses, name="mean_loss") + l2_reg_lambda * l2_loss + sum(
                                                              tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.sign(self.probs - 0.5), tf.sign(self.target - 0.5))    # [batch_size, ]
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
