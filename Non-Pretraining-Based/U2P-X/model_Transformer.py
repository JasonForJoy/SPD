import tensorflow as tf
import numpy as np
import transformer_block

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


class Transformer(object):
    def __init__(
      self, max_utter_num, max_utter_len, max_profile_num, max_profile_len, num_layer, vocab_size, embedding_size, vocab, rnn_size, maxWordLength, charVocab, l2_reg_lambda=0.0):

        self.utterances = tf.placeholder(tf.int32, [None, max_utter_num, max_utter_len], name="utterances")
        self.utterances_len = tf.placeholder(tf.int32, [None, max_utter_num], name="utterances_len")
        self.utterances_num = tf.placeholder(tf.int32, [None], name="utterances_num")
        self.profiles = tf.placeholder(tf.int32, [None, max_profile_num, max_profile_len], name="profiles")
        self.profiles_len = tf.placeholder(tf.int32, [None, max_profile_num], name="profiles_len")
        self.profiles_num = tf.placeholder(tf.int32, [None], name="profiles_num")

        self.target = tf.placeholder(tf.float32, [None], name="target")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.u_charVec = tf.placeholder(tf.int32, [None, max_utter_num, max_utter_len, maxWordLength], name="utterances_char")
        self.u_charLen = tf.placeholder(tf.int32, [None, max_utter_num, max_utter_len], name="utterances_char_len")
        self.p_charVec = tf.placeholder(tf.int32, [None, max_profile_num, max_profile_len, maxWordLength], name="profiles_char")
        self.p_charLen =  tf.placeholder(tf.int32, [None, max_profile_num, max_profile_len], name="profiles_char_len")

        l2_loss = tf.constant(1.0)


        # =============================== Embedding layer ===============================
        with tf.name_scope("embedding"):
            W = get_embeddings(vocab)
            utterances_embedded = tf.nn.embedding_lookup(W, self.utterances)  # [batch_size, max_utter_num, max_utter_len,  word_dim]
            profiles_embedded = tf.nn.embedding_lookup(W, self.profiles)      # [batch_size, max_profile_num, max_profile_len, word_dim]
            utterances_embedded = tf.nn.dropout(utterances_embedded, keep_prob=self.dropout_keep_prob)
            profiles_embedded = tf.nn.dropout(profiles_embedded, keep_prob=self.dropout_keep_prob)
            print("utterances_embedded: {}".format(utterances_embedded.get_shape()))
            print("profiles_embedded: {}".format(profiles_embedded.get_shape()))


        # =============================== Encoding layer ===============================
        with tf.variable_scope("encoding_layer") as vs:
            rnn_scope_name = "bidirectional_rnn"
            emb_dim = utterances_embedded.get_shape()[-1].value
            flattened_utterances_embedded = tf.reshape(utterances_embedded, [-1, max_utter_len, emb_dim])  # [batch_size*max_utter_num, max_utter_len, emb]
            flattened_utterances_len = tf.reshape(self.utterances_len, [-1])                               # [batch_size*max_utter_num, ]
            flattened_profiles_embedded = tf.reshape(profiles_embedded, [-1, max_profile_len, emb_dim])    # [batch_size*max_profile_num, max_profile_len, emb]
            flattened_profiles_len = tf.reshape(self.profiles_len, [-1])                                   # [batch_size*max_profile_num, ]

            utterances_input = flattened_utterances_embedded
            profiles_input = flattened_profiles_embedded
            for layer in range(num_layer):
                with tf.variable_scope("encoding_layer_{}".format(layer)):
                    utterances_output = transformer_block.block(utterances_input, utterances_input, utterances_input, 
                                                            flattened_utterances_len, flattened_utterances_len)
                    utterances_input = utterances_output

            for layer in range(num_layer):
                with tf.variable_scope("encoding_layer_{}".format(layer), reuse=True):
                    profiles_output = transformer_block.block(profiles_input, profiles_input, profiles_input, 
                                                            flattened_profiles_len, flattened_profiles_len)
                    profiles_input = profiles_output
            print("establish Transformer encoder")
            print("utterances_output: {}".format(utterances_output.get_shape()))
            print("profiles_output: {}".format(profiles_output.get_shape()))
            

        # =============================== Matching layer ===============================
        with tf.variable_scope("matching_layer") as vs:
            mask_u = tf.sequence_mask(flattened_utterances_len, max_utter_len, dtype=tf.float32)  # [batch_size*max_utter_num, max_utter_len]
            utterances_output = utterances_output * tf.expand_dims(mask_u, 2)                     # [batch_size*max_utter_num, max_utter_len, dim]
            final_utterances = tf.reduce_sum(utterances_output, axis=1)                           # [batch_size*max_utter_num, dim]
            # final_utterances = tf.div(final_utterances, tf.expand_dims(tf.sqrt(tf.cast(flattened_utterances_len, tf.float32)), 1))
            concat_dim = final_utterances.get_shape()[-1].value
            final_utterances = tf.reshape(final_utterances, [-1, max_utter_num, concat_dim])      # [batch_size, max_utter_num, dim]

            mask_p = tf.sequence_mask(flattened_profiles_len, max_profile_len, dtype=tf.float32)  # [batch_size*max_profile_num, max_profile_len]
            profiles_output = profiles_output * tf.expand_dims(mask_p, 2)
            final_profiles = tf.reduce_sum(profiles_output, axis=1)
            # final_profiles = tf.div(final_profiles, tf.expand_dims(tf.sqrt(tf.cast(flattened_profiles_len, tf.float32)), 1))
            final_profiles = tf.reshape(final_profiles, [-1, max_profile_num, concat_dim])        # [batch_size, max_profile_num, dim]

            A_matrix = tf.get_variable('A_matrix_v', shape=[concat_dim, concat_dim], initializer=tf.orthogonal_initializer(), dtype=tf.float32)
            similarity = tf.einsum('aij,jk->aik', 
                                   final_utterances, A_matrix)   # [batch_size, max_utter_num, dim]
            similarity = tf.matmul(similarity, 
                                   tf.transpose(final_profiles, perm=[0, 2, 1]))  # [batch_size, max_utter_num, max_profile_num]

            print("shape of similarity: {}".format(similarity.get_shape()))


        # =============================== Aggregation layer ===============================
        with tf.variable_scope("aggregation_layer") as vs:
            logits = tf.reduce_max(similarity, axis=2)  # [batch_size, max_utter_num]
            mask_u = tf.sequence_mask(self.utterances_num, max_utter_num, dtype=tf.float32)  # [batch_size, max_utter_num]
            logits = logits * mask_u
            logits = tf.reduce_sum(logits, axis=1)      # [batch_size, ]
            print("establish reduce_max across profiles and masked_reduce_sum across utterances")
            print("logits: {}".format(logits.get_shape()))


        # =============================== Prediction layer ===============================
        with tf.variable_scope("prediction_layer") as vs:
            self.probs = tf.sigmoid(logits, name="prob")   # [batch_size, ]
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.target)
            self.mean_loss = tf.reduce_mean(losses, name="mean_loss") + l2_reg_lambda * l2_loss + sum(
                                                              tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.sign(self.probs - 0.5), tf.sign(self.target - 0.5))    # [batch_size, ]
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
