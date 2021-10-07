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

def lstm_layer(inputs, input_seq_len, rnn_size, dropout_keep_prob, scope, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse) as vs:
        fw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        fw_cell  = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_prob)
        bw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        bw_cell  = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_prob)
        rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                                inputs=inputs,
                                                                sequence_length=input_seq_len,
                                                                dtype=tf.float32)
        return rnn_outputs, rnn_states

def multi_lstm_layer(inputs, input_seq_len, rnn_size, dropout_keep_prob, num_layer, scope, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse) as vs:
        multi_outputs = []
        multi_states = []
        cur_inputs = inputs
        for i_layer in range(num_layer):
            rnn_outputs, rnn_states = lstm_layer(cur_inputs, input_seq_len, rnn_size, dropout_keep_prob, scope+str(i_layer), scope_reuse)
            rnn_outputs = tf.concat(values=rnn_outputs, axis=2)
            multi_outputs.append(rnn_outputs)
            multi_states.append(rnn_states)
            cur_inputs = rnn_outputs

        # multi_layer_aggregation
        ml_weights = tf.nn.softmax(tf.get_variable("ml_scores", [num_layer, ], initializer=tf.constant_initializer(0.0)))

        multi_outputs = tf.stack(multi_outputs, axis=-1)   # [batch_size, max_len, 2*rnn_size(400), num_layer]
        max_len = multi_outputs.get_shape()[1].value
        dim = multi_outputs.get_shape()[2].value
        flattened_multi_outputs = tf.reshape(multi_outputs, [-1, num_layer])                         # [batch_size * max_len * 2*rnn_size(400), num_layer]
        aggregated_ml_outputs = tf.matmul(flattened_multi_outputs, tf.expand_dims(ml_weights, 1))    # [batch_size * max_len * 2*rnn_size(400), 1]
        aggregated_ml_outputs = tf.reshape(aggregated_ml_outputs, [-1, max_len, dim])                # [batch_size , max_len , 2*rnn_size(400)]

        return aggregated_ml_outputs

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

def context_response_similarity_matrix(context, response):
    # q_len = context.get_shape()[1].value
    # r_len = response.get_shape()[1].value
    # dim = context.get_shape()[2].value

    # response : batch_size * r_len * dim
    # [batch_size, dim, q_len]
    c2 = tf.transpose(context, perm=[0,2,1])

    # [batch_size, response_len, max_utter_num*max_utter_len]
    similarity = tf.matmul(response, c2, name='similarity_matrix')

    return similarity

def attended_response(similarity_matrix, context, flattened_utters_len, max_utter_len, max_utter_num):
    # similarity_matrix:    [batch_size, max_response_num*response_len, max_utter_num*max_utter_len]
    # context:              [batch_size, max_utter_num*max_utter_len, dim]
    # flattened_utters_len: [batch_size* max_utter_num, ]
    
    # masked similarity_matrix
    mask_c = tf.sequence_mask(flattened_utters_len, max_utter_len, dtype=tf.float32)  # [batch_size*max_utter_num, max_utter_len]
    mask_c = tf.reshape(mask_c, [-1, max_utter_num*max_utter_len])                    # [batch_size, max_utter_num*max_utter_len]
    mask_c = tf.expand_dims(mask_c, 1)                                                # [batch_size, 1, max_utter_num*max_utter_len]
    similarity_matrix = similarity_matrix * mask_c + -1e9 * (1-mask_c)                # [batch_size, max_response_num*response_len, max_utter_num*max_utter_len]

    attention_weight_for_c = tf.nn.softmax(similarity_matrix, dim=-1)  # [batch_size, max_response_num*response_len, max_utter_num*max_utter_len]
    attended_response = tf.matmul(attention_weight_for_c, context)     # [batch_size, max_response_num*response_len, dim]

    return attended_response

def attended_context(similarity_matrix, response, flattened_responses_len, max_response_len, max_response_num):
    # similarity_matrix:    [batch_size, max_response_num*response_len, max_utter_num*max_utter_len]
    # response:             [batch_size, max_response_num*response_len, dim]
    # flattened_utters_len: [batch_size* max_response_num, ]

    # masked similarity_matrix
    mask_r = tf.sequence_mask(flattened_responses_len, max_response_len, dtype=tf.float32)  # [batch_size*max_response_num, response_len]
    mask_r = tf.reshape(mask_r, [-1, max_response_num*max_response_len])                    # [batch_size, max_response_num*response_len]
    mask_r = tf.expand_dims(mask_r, 2)                                           # [batch_size, max_response_num*response_len, 1]
    similarity_matrix = similarity_matrix * mask_r + -1e9 * (1-mask_r)           # [batch_size, max_response_num*response_len, max_utter_num*max_utter_len]

    attention_weight_for_r = tf.nn.softmax(tf.transpose(similarity_matrix, perm=[0,2,1]), dim=-1)  # [batch_size, max_utter_num*max_utter_len, max_response_num*response_len]
    attended_context = tf.matmul(attention_weight_for_r, response)                                 # [batch_size, max_utter_num*max_utter_len, dim]
    
    return attended_context


class ESIM(object):
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
        # word embedding
        with tf.name_scope("embedding"):
            W = get_embeddings(vocab)
            utterances_embedded = tf.nn.embedding_lookup(W, self.utterances)  # [batch_size, max_utter_num, max_utter_len,  word_dim]
            profiles_embedded = tf.nn.embedding_lookup(W, self.profiles)      # [batch_size, max_profile_num, max_profile_len, word_dim]
            print("original utterances_embedded: {}".format(utterances_embedded.get_shape()))
            print("original profiles_embedded: {}".format(profiles_embedded.get_shape()))

        # char CNN
        # with tf.name_scope('char_embedding'):
        #     char_W = get_char_embedding(charVocab)
        #     utterances_char_embedded = tf.nn.embedding_lookup(char_W, self.u_charVec)  # [batch_size, max_utter_num, max_utter_len,  maxWordLength, char_dim]
        #     profiles_char_embedded   = tf.nn.embedding_lookup(char_W, self.p_charVec)  # [batch_size, max_profile_num, max_profile_len, maxWordLength, char_dim]
        #     print("utterances_char_embedded: {}".format(utterances_char_embedded.get_shape()))
        #     print("profiles_char_embedded: {}".format(profiles_char_embedded.get_shape()))

        # char_dim = utterances_char_embedded.get_shape()[-1].value
        # utterances_char_embedded = tf.reshape(utterances_char_embedded, [-1, maxWordLength, char_dim])  # [batch_size*max_utter_num*max_utter_len, maxWordLength, char_dim]
        # profiles_char_embedded = tf.reshape(profiles_char_embedded, [-1, maxWordLength, char_dim])      # [batch_size*max_profile_num*max_profile_len, maxWordLength, char_dim]

        # utterances_cnn_char_emb = cnn_layer(utterances_char_embedded, filter_sizes=[3, 4, 5], num_filters=50, scope="CNN_char_emb", scope_reuse=False) # [batch_size*max_utter_num*max_utter_len,   emb]
        # cnn_char_dim = utterances_cnn_char_emb.get_shape()[1].value
        # utterances_cnn_char_emb = tf.reshape(utterances_cnn_char_emb, [-1, max_utter_num, max_utter_len, cnn_char_dim])                                # [batch_size, max_utter_num, max_utter_len, emb]

        # profiles_cnn_char_emb = cnn_layer(profiles_char_embedded, filter_sizes=[3, 4, 5], num_filters=50, scope="CNN_char_emb", scope_reuse=True)      # [batch_size*max_profile_len,  cnn_char_dim]
        # profiles_cnn_char_emb = tf.reshape(profiles_cnn_char_emb, [-1, max_profile_num, max_profile_len, cnn_char_dim])                                # [batch_size, max_profile_len, cnn_char_dim]
                
        # utterances_embedded = tf.concat(axis=-1, values=[utterances_embedded, utterances_cnn_char_emb])   # [batch_size, max_utter_num, max_utter_len, emb]
        # profiles_embedded  = tf.concat(axis=-1, values=[profiles_embedded, profiles_cnn_char_emb])        # [batch_size, max_profile_num, max_profile_len, emb]
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
            flattened_profiles_len = tf.reshape(self.profiles_len, [-1])                                  # [batch_size*max_profile_num, ]
            # 1. single_lstm_layer
            u_rnn_output, u_rnn_states = lstm_layer(flattened_utterances_embedded, flattened_utterances_len, rnn_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=False)
            utterances_output = tf.concat(axis=2, values=u_rnn_output)   # [batch_size*max_utter_num,  max_utter_len, rnn_size*2]
            p_rnn_output, p_rnn_states = lstm_layer(flattened_profiles_embedded, flattened_profiles_len, rnn_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=True)   # [batch_size, max_profile_len, rnn_size(200)]
            profiles_output = tf.concat(axis=2, values=p_rnn_output)     # [batch_size*max_profile_num, max_profile_len, 2*rnn_size(400)]
            # 2. multi_lstm_layer
            # utterances_output = multi_lstm_layer(flattened_utterances_embedded, flattened_utterances_len, rnn_size, self.dropout_keep_prob, num_layer, rnn_scope_name, scope_reuse=False)
            # response_output = multi_lstm_layer(flattened_responses_embedded, flattened_responses_len, rnn_size, self.dropout_keep_prob, num_layer, rnn_scope_name, scope_reuse=True)
            # print("establish AHRE layers : {}".format(num_layer))

        # =============================== Matching layer ===============================
        with tf.variable_scope("matching_layer") as vs:
            output_dim = utterances_output.get_shape()[-1].value
            utterances_output = tf.reshape(utterances_output, [-1, max_utter_num*max_utter_len, output_dim])  # [batch_size, max_utter_num*max_utter_len, rnn_size*2]
            profiles_output = tf.reshape(profiles_output, [-1, max_profile_num*max_profile_len, output_dim])  # [batch_size, max_profile_num*max_profile_len, rnn_size*2]

            similarity = context_response_similarity_matrix(utterances_output, profiles_output)  # [batch_size, max_profile_num*max_profile_len, max_utter_num*max_utter_len]
            attended_utterances_output = attended_context(similarity, profiles_output, flattened_profiles_len, max_profile_len, max_profile_num)  # [batch_size, max_utter_num*max_utter_len, dim]
            attended_profiles_output = attended_response(similarity, utterances_output, flattened_utterances_len, max_utter_len, max_utter_num)   # [batch_size, max_profile_num*max_profile_len, dim]
            
            m_u = tf.concat(axis=2, values=[utterances_output, attended_utterances_output, tf.multiply(utterances_output, attended_utterances_output), utterances_output-attended_utterances_output])   # [batch_size, max_utter_num*max_utter_len, dim]
            m_p = tf.concat(axis=2, values=[profiles_output, attended_profiles_output, tf.multiply(profiles_output, attended_profiles_output), profiles_output-attended_profiles_output])               # [batch_size, max_profile_num*max_profile_len, dim]
            concat_dim = m_u.get_shape()[-1].value
            m_u = tf.reshape(m_u, [-1, max_utter_len, concat_dim])    # [batch_size*max_utter_num, max_utter_len, dim]
            m_p = tf.reshape(m_p, [-1, max_profile_len, concat_dim])  # [batch_size*max_profile_num, max_profile_len, dim]
            print("establish matching between utterances and profiles")


        # =============================== Aggregation layer ===============================
        with tf.variable_scope("aggregation_layer") as vs:
            # context RNN aggregation
            rnn_scope_cross = 'bidirectional_rnn_cross'
            u_rnn_output_2, u_rnn_state_2 = lstm_layer(m_u, flattened_utterances_len, rnn_size, self.dropout_keep_prob, rnn_scope_cross, scope_reuse=False)
            utterances_output_cross = tf.concat(axis=2, values=u_rnn_output_2)   # [batch_size*max_utter_num, max_utter_len,  rnn_size*2]
            final_utterances_max = tf.reduce_max(utterances_output_cross, axis=1)
            final_utterances_state = tf.concat(axis=1, values=[u_rnn_state_2[0].h, u_rnn_state_2[1].h])
            final_utterances = tf.concat(axis=1, values=[final_utterances_max, final_utterances_state])
            final_utterances = tf.reshape(final_utterances, [-1, max_utter_num, output_dim*2])   # [batch_size, max_utter_num, 4*rnn_size]
            
            rnn_scope_aggre = "bidirectional_rnn_aggregation"
            final_utterances_output, final_utterances_state = lstm_layer(final_utterances, self.utterances_num, rnn_size, self.dropout_keep_prob, rnn_scope_aggre, scope_reuse=False)
            final_utterances_output = tf.concat(axis=2, values=final_utterances_output)   # [batch_size, max_utter_num, 2*rnn_size]
            final_utterances_max = tf.reduce_max(final_utterances_output, axis=1)         # [batch_size, 2*rnn_size]
            final_utterances_state = tf.concat(axis=1, values=[final_utterances_state[0].h, final_utterances_state[1].h])  # [batch_size, 2*rnn_size]
            final_utterances = tf.concat(axis=1, values=[final_utterances_max, final_utterances_state])
            print("establish RNN aggregation on utterances")

            # Persona self projection attention aggregation
            p_rnn_output_2, p_rnn_state_2 = lstm_layer(m_p, flattened_profiles_len, rnn_size, self.dropout_keep_prob, rnn_scope_cross, scope_reuse=True)
            profiles_output_cross = tf.concat(axis=2, values=p_rnn_output_2)     # [batch_size, max_profile_len, rnn_size*2]
            final_profiles_max = tf.reduce_max(profiles_output_cross, axis=1)
            final_profiles_state = tf.concat(axis=1, values=[p_rnn_state_2[0].h, p_rnn_state_2[1].h])
            final_profiles = tf.concat(axis=1, values=[final_profiles_max, final_profiles_state])
            final_profiles = tf.reshape(final_profiles, [-1, max_profile_num, output_dim*2])   # [batch_size, max_profile_num, 4*rnn_size]
            
            proj_W_profiles = tf.get_variable("proj_W_profiles", [output_dim*2, 1], initializer=tf.orthogonal_initializer())
            proj_b_profiles = tf.get_variable("proj_b_profiles", [1, ], initializer=tf.constant_initializer(0.0))
            profiles_weights = tf.einsum('bij,jk->bik', final_profiles, proj_W_profiles) + proj_b_profiles  # [batch_size, max_profile_num, 1]
            profiles_weights = tf.squeeze(profiles_weights, [-1])                                           # [batch_size, max_profile_num]
            profiles_mask = tf.sequence_mask(self.profiles_num, max_profile_num, dtype=tf.float32)          # [batch_size, max_profile_num]
            profiles_weights = tf.nn.softmax(profiles_weights * profiles_mask + -1e9 * (1-profiles_mask))   # [batch_size, max_profile_num]
            final_profiles = tf.matmul(tf.transpose(final_profiles, perm=[0,2,1]),  # [batch_size, dim, max_profile_num]
                                       tf.expand_dims(profiles_weights, -1))        # [batch_size, max_profile_num, 1]  ==> [batch_size, dim, 1]
            final_profiles = tf.squeeze(final_profiles, [-1])                       # [batch_size, dim]
            final_profiles = tf.nn.dropout(final_profiles, keep_prob=self.dropout_keep_prob)
            print("establish attention aggregation on profiles")

            joined_feature = tf.concat(axis=1, values=[final_utterances, final_profiles])  # [batch_size, 8*rnn_size(1600)]
            print("joined feature: {}".format(joined_feature.get_shape()))            


        # =============================== Prediction layer ===============================
        with tf.variable_scope("prediction_layer") as vs:
            hidden_input_size = joined_feature.get_shape()[1].value
            hidden_output_size = 256
            regularizer = tf.contrib.layers.l2_regularizer(l2_reg_lambda)
            # dropout On MLP
            joined_feature = tf.nn.dropout(joined_feature, keep_prob=self.dropout_keep_prob)
            full_out = tf.contrib.layers.fully_connected(joined_feature, hidden_output_size,
                                                            activation_fn=tf.nn.relu,
                                                            reuse=False,
                                                            trainable=True,
                                                            scope="projected_layer")   # [batch_size, hidden_output_size(256)]
            full_out = tf.nn.dropout(full_out, keep_prob=self.dropout_keep_prob)

            last_weight_dim = full_out.get_shape()[1].value
            print("last_weight_dim: {}".format(last_weight_dim))
            bias = tf.Variable(tf.constant(0.1, shape=[1]), name="bias")
            s_w = tf.get_variable("s_w", shape=[last_weight_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.matmul(full_out, s_w) + bias   # [batch_size, 1]
            print("logits: {}".format(logits.get_shape()))
            
            logits = tf.squeeze(logits, [1])   # [batch_size, ]
            self.probs = tf.sigmoid(logits, name="prob")   # [batch_size, ]

            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.target)
            self.mean_loss = tf.reduce_mean(losses, name="mean_loss") + l2_reg_lambda * l2_loss + sum(
                                                              tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.sign(self.probs - 0.5), tf.sign(self.target - 0.5))    # [batch_size, ]
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
