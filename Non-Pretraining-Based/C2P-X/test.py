import tensorflow as tf
import numpy as np
import os
import time
import datetime
import operator
import metrics
from collections import defaultdict
import data_helpers

# Files
tf.flags.DEFINE_string("test_file", "", "path to test file")
tf.flags.DEFINE_string("vocab_file", "", "vocabulary file")
tf.flags.DEFINE_string("char_vocab_file", "", "vocabulary file")
tf.flags.DEFINE_string("output_file", "", "prediction output file")

# Model Hyperparameters
tf.flags.DEFINE_integer("max_context_len", 150, "max context length")
tf.flags.DEFINE_integer("max_persona_len", 50, "max persona length")
tf.flags.DEFINE_integer("max_word_length", 18, "max word length")

# Test parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
print("")

vocab = data_helpers.load_vocab(FLAGS.vocab_file)
print('vocabulary size: {}'.format(len(vocab)))
charVocab = data_helpers.load_char_vocab(FLAGS.char_vocab_file)

test_dataset = data_helpers.load_dataset(FLAGS.test_file, vocab, FLAGS.max_context_len, FLAGS.max_persona_len)
print('test_pairs: {}'.format(len(test_dataset)))

print("\nEvaluating...\n")

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print(checkpoint_file)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        context     = graph.get_operation_by_name("context").outputs[0]
        context_len = graph.get_operation_by_name("context_len").outputs[0]
        persona     = graph.get_operation_by_name("persona").outputs[0]
        persona_len = graph.get_operation_by_name("persona_len").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        c_char_feature = graph.get_operation_by_name("context_char").outputs[0]
        c_char_len     = graph.get_operation_by_name("context_char_len").outputs[0]
        p_char_feature = graph.get_operation_by_name("persona_char").outputs[0]
        p_char_len     = graph.get_operation_by_name("persona_char_len").outputs[0]

        # Tensors we want to evaluate
        prob = graph.get_operation_by_name("prediction_layer/prob").outputs[0]

        results = defaultdict(list)
        num_test = 0

        test_batches = data_helpers.batch_iter(test_dataset, FLAGS.batch_size, 1, FLAGS.max_context_len, FLAGS.max_persona_len, charVocab, FLAGS.max_word_length, shuffle=False)
        for test_batch in test_batches:
            x_context, x_context_len, x_persona, x_persona_len, x_labels, x_id_pairs, x_context_char, x_context_char_len, x_persona_char, x_persona_char_len = test_batch
            feed_dict = {
                context: x_context,
                context_len: x_context_len,
                persona: x_persona,
                persona_len: x_persona_len,
                dropout_keep_prob: 1.0,
                c_char_feature: x_context_char,
                c_char_len: x_context_char_len,
                p_char_feature: x_persona_char,
                p_char_len: x_persona_char_len
            }
            predicted_prob = sess.run(prob, feed_dict)
            num_test += len(predicted_prob)
            print('num_test_sample={}'.format(num_test))
            for i, prob_score in enumerate(predicted_prob):
                us_id, ps_id, label = x_id_pairs[i]
                results[us_id].append((ps_id, label, prob_score))

accu, precision, recall, f1, loss = metrics.classification_metrics(results)
print('Accuracy: {}, Precision: {}  Recall: {}  F1: {} Loss: {}'.format(accu, precision, recall, f1, loss))

mvp = metrics.mean_average_precision(results)
mrr = metrics.mean_reciprocal_rank(results)
top_1_precision = metrics.top_1_precision(results)
total_valid_query = metrics.get_num_valid_query(results)
print('MAP (mean average precision: {}\tMRR (mean reciprocal rank): {}\tTop-1 precision: {}\tNum_query: {}'.format(mvp, mrr, top_1_precision, total_valid_query))

out_path = FLAGS.output_file
print("Saving evaluation to {}".format(out_path))
with open(out_path, 'w') as f:
    f.write("query_id\tdocument_id\tscore\trank\trelevance\n")
    for us_id, v in results.items():
        v.sort(key=operator.itemgetter(2), reverse=True)
        for i, rec in enumerate(v):
            ps_id, label, prob_score = rec
            rank = i+1
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(us_id, ps_id, prob_score, rank, label))
