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

def cnn_layer(inputs, filter_sizes, num_filters, scope=None, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse):
        input_size = inputs.get_shape()[2].value

        outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv_{}".format(i)):
                w = tf.get_variable("w", [filter_size, input_size, num_filters])
                b = tf.get_variable("b", [num_filters])
            conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID") # [num_words, num_chars - filter_size, num_filters]
            h = tf.nn.relu(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_size, num_filters]
            pooled = tf.reduce_max(h, 1) # [num_words, num_filters]
            outputs.append(pooled)
    return tf.concat(outputs, 1) # [num_words, num_filters * len(filter_sizes)]


class Transformer(object):
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
        # 1. word embedding
        with tf.name_scope("embedding"):
            W = get_embeddings(vocab)
            context_embedded = tf.nn.embedding_lookup(W, self.context)  # [batch_size, max_context_len, word_dim]
            persona_embedded = tf.nn.embedding_lookup(W, self.persona)  # [batch_size, max_persona_len, word_dim]
            context_embedded = tf.nn.dropout(context_embedded, keep_prob=self.dropout_keep_prob)
            persona_embedded = tf.nn.dropout(persona_embedded, keep_prob=self.dropout_keep_prob)
            print("context_embedded: {}".format(context_embedded.get_shape()))
            print("persona_embedded: {}".format(persona_embedded.get_shape()))


        # =============================== Encoding layer ===============================
        emb_dim = context_embedded.get_shape()[-1].value
        
        # with tf.variable_scope("encoding_layer") as vs:
        #     # CNN encoder
        #     final_context = cnn_layer(context_embedded, filter_sizes=[3, 4, 5], num_filters=100, scope="CNN_emb", scope_reuse=False) # [batch_size*max_utter_num, emb]
        #     final_persona = cnn_layer(persona_embedded, filter_sizes=[3, 4, 5], num_filters=100, scope="CNN_emb", scope_reuse=True)  # [batch_size*max_profile_num, emb]
        #     print("establish CNN encoder")

        context_input = context_embedded
        for layer in range(num_layer):
            with tf.variable_scope("encoding_layer_{}".format(layer)):
                context_output = transformer_block.block(context_input, context_input, context_input, self.context_len, self.context_len)
                context_input = context_output

        persona_input = persona_embedded
        for layer in range(num_layer):
            with tf.variable_scope("encoding_layer_{}".format(layer), reuse=True):   # [batch_size, max_context_len, word_dim]
                persona_output = transformer_block.block(persona_input, persona_input, persona_input, self.persona_len, self.persona_len)
                persona_input = persona_output
        print("context_output: {}".format(context_output.get_shape()))  # [batch_size, max_persona_len, word_dim]
        print("persona_output: {}".format(persona_output.get_shape()))
        print("establish {}-layer Transformer encoder".format(num_layer))


        # =============================== Matching layer ===============================
        with tf.variable_scope("matching_layer") as vs:
            mask_c = tf.sequence_mask(self.context_len, max_context_len, dtype=tf.float32)  # [batch_size, max_context_len]
            context_output = context_output * tf.expand_dims(mask_c, 2)                     # [batch_size, max_context_len, dim]
            final_context = tf.reduce_sum(context_output, axis=1)                           # [batch_size, dim]
            
            mask_p = tf.sequence_mask(self.persona_len, max_persona_len, dtype=tf.float32)  # [batch_size, max_persona_len]
            persona_output = persona_output * tf.expand_dims(mask_p, 2)                     # [batch_size, max_persona_len, dim]
            final_persona = tf.reduce_sum(persona_output, axis=1)                           # [batch_size, dim]

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
