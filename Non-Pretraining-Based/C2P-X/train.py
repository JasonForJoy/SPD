import tensorflow as tf
import numpy as np
import os
import time
import datetime
import operator
from collections import defaultdict
import metrics
import data_helpers
from model_BOW import BOW as MODEL
# from model_BiLSTM import BiLSTM as MODEL
# from model_Transformer import Transformer as MODEL
# from model_ESIM import ESIM as MODEL


# Files
tf.flags.DEFINE_string("train_file", "", "path to train file")
tf.flags.DEFINE_string("valid_file", "", "path to valid file")
tf.flags.DEFINE_string("vocab_file", "", "vocabulary file")
tf.flags.DEFINE_string("char_vocab_file",  "", "path to char vocab file")
tf.flags.DEFINE_string("embedded_vector_file", "", "pre-trained embedded word vector")

# Model Hyperparameters
tf.flags.DEFINE_integer("max_context_len", 150, "max context length")
tf.flags.DEFINE_integer("max_persona_len", 50, "max persona length")
tf.flags.DEFINE_integer("max_word_length", 18, "max word length")
tf.flags.DEFINE_integer("num_layer", 1, "num of layers in sentence encoder")
tf.flags.DEFINE_integer("embedding_dim", 200, "dimensionality of word embedding")
tf.flags.DEFINE_integer("rnn_size", 200, "number of RNN units")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "batch size (default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "dropout keep probability (default: 1.0)")
tf.flags.DEFINE_integer("num_epochs", 1000000, "number of training epochs (default: 1000000)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "evaluate model on valid dataset after this many steps (default: 1000)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
print("")

# Load data
print("Loading data...")

vocab = data_helpers.load_vocab(FLAGS.vocab_file)
print('vocabulary size: {}'.format(len(vocab)))
charVocab = data_helpers.load_char_vocab(FLAGS.char_vocab_file)

train_dataset = data_helpers.load_dataset(FLAGS.train_file, vocab, FLAGS.max_context_len, FLAGS.max_persona_len)
print('train_pairs: {}'.format(len(train_dataset)))
valid_dataset = data_helpers.load_dataset(FLAGS.valid_file, vocab, FLAGS.max_context_len, FLAGS.max_persona_len)
print('valid_pairs: {}'.format(len(valid_dataset)))

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        model = MODEL(
            max_context_len=FLAGS.max_context_len,
            max_persona_len=FLAGS.max_persona_len,
            num_layer=FLAGS.num_layer,
            vocab_size=len(vocab),
            embedding_size=FLAGS.embedding_dim,
            vocab=vocab,
            rnn_size=FLAGS.rnn_size,
            maxWordLength=FLAGS.max_word_length,
            charVocab=charVocab,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                                   5000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.mean_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        """
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)
        """

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        out_dir = os.path.abspath(os.path.join("../output", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        """
        loss_summary = tf.scalar_summary("loss", model.mean_loss)
        acc_summary = tf.scalar_summary("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)
        """

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_context, x_context_len, x_persona, x_persona_len, x_labels, x_id_pairs, 
                       x_context_char, x_context_char_len, x_persona_char, x_persona_char_len):
            """
            A single training step
            """
            feed_dict = {
              model.context: x_context,
              model.context_len: x_context_len,
              model.persona: x_persona,
              model.persona_len: x_persona_len,
              model.target: x_labels,
              model.dropout_keep_prob: FLAGS.dropout_keep_prob,
              model.c_charVec: x_context_char,
              model.c_charLen: x_context_char_len,
              model.p_charVec: x_persona_char,
              model.p_charLen: x_persona_char_len
            }

            _, step, loss, accuracy, predicted_prob = sess.run(
                [train_op, global_step, model.mean_loss, model.accuracy, model.probs],
                feed_dict)

            if step%100 == 0:
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #train_summary_writer.add_summary(summaries, step)


        def dev_step():
            results = defaultdict(list)
            num_test = 0
            num_correct = 0.0
            valid_batches = data_helpers.batch_iter(valid_dataset, FLAGS.batch_size, 1, FLAGS.max_context_len, FLAGS.max_persona_len, charVocab, FLAGS.max_word_length, shuffle=True)
            for valid_batch in valid_batches:
                x_context, x_context_len, x_persona, x_persona_len, x_labels, x_id_pairs, x_context_char, x_context_char_len, x_persona_char, x_persona_char_len = valid_batch
                feed_dict = {
                  model.context: x_context,
                  model.context_len: x_context_len,
                  model.persona: x_persona,
                  model.persona_len: x_persona_len,
                  model.target: x_labels,
                  model.dropout_keep_prob: 1.0,
                  model.c_charVec: x_context_char,
                  model.c_charLen: x_context_char_len,
                  model.p_charVec: x_persona_char,
                  model.p_charLen: x_persona_char_len
                }
                batch_accuracy, predicted_prob = sess.run([model.accuracy, model.probs], feed_dict)
                num_test += len(predicted_prob)
                if num_test % 1000 == 0:
                    print(num_test)

                num_correct += len(predicted_prob) * batch_accuracy
                for i, prob_score in enumerate(predicted_prob):
                    utterances_id, profiles_id, label = x_id_pairs[i]
                    results[utterances_id].append((profiles_id, label, prob_score))

            #calculate top-1 precision
            print('num_test_samples: {}  test_accuracy: {}'.format(num_test, num_correct/num_test))
            accu, precision, recall, f1, loss = metrics.classification_metrics(results)
            print('Accuracy: {}, Precision: {}  Recall: {}  F1: {} Loss: {}'.format(accu, precision, recall, f1, loss))

            mvp = metrics.mean_average_precision(results)
            mrr = metrics.mean_reciprocal_rank(results)
            top_1_precision = metrics.top_1_precision(results)
            total_valid_query = metrics.get_num_valid_query(results)
            print('MAP (mean average precision: {}\tMRR (mean reciprocal rank): {}\tTop-1 precision: {}\tNum_query: {}'.format(mvp, mrr, top_1_precision, total_valid_query))

            return mrr

        best_mrr = 0.0
        batches = data_helpers.batch_iter(train_dataset, FLAGS.batch_size, FLAGS.num_epochs, FLAGS.max_context_len, FLAGS.max_persona_len, charVocab, FLAGS.max_word_length, shuffle=True)
        for batch in batches:
            x_context, x_context_len, x_persona, x_persona_len, x_labels, x_id_pairs, x_context_char, x_context_char_len, x_persona_char, x_persona_char_len = batch
            train_step(x_context, x_context_len, x_persona, x_persona_len, x_labels, x_id_pairs, x_context_char, x_context_char_len, x_persona_char, x_persona_char_len)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                valid_mrr = dev_step()
                if valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
    