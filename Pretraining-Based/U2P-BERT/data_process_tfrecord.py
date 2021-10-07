# -*- coding: utf-8 -*-
import os
import collections
import tensorflow as tf
from tqdm import tqdm
import tokenization


'''
modify valid_file/test_file with 
valid_both_revised_cand_100.txt/test_both_revised_cand_100.txt
for another evaluation setting
'''
tf.flags.DEFINE_string("train_file", "../../data_PMPC/train_both_revised.txt", 
    "path to train file")
tf.flags.DEFINE_string("valid_file", "../../data_PMPC/valid_both_revised_cand_10.txt", 
    "path to valid file")
tf.flags.DEFINE_string("test_file", "../../data_PMPC/test_both_revised_cand_10.txt", 
    "path to test file")

tf.flags.DEFINE_string("vocab_file", "../uncased_L-12_H-768_A-12/vocab.txt", 
    "path to vocab file")
tf.flags.DEFINE_integer("max_seq_length", 200, 
	"max sequence length of concatenated context and response")
tf.flags.DEFINE_bool("do_lower_case", True,
    "whether to lower case the input text")

max_sentence_a_num=8
max_sentence_a_len=20
max_sentence_b_num=5
max_sentence_b_len=15


def print_configuration_op(FLAGS):
    print('My Configurations:')
    for name, value in FLAGS.__flags.items():
        value=value.value
        if type(value) == float:
            print(' %s:\t %f'%(name, value))
        elif type(value) == int:
            print(' %s:\t %d'%(name, value))
        elif type(value) == str:
            print(' %s:\t %s'%(name, value))
        elif type(value) == bool:
            print(' %s:\t %s'%(name, value))
        else:
            print('%s:\t %s' % (name, value))
    print('End of configuration')


def load_dataset(fname, tfrecord_path="data_tfrecord/"):

    if not os.path.exists(tfrecord_path):
        os.makedirs(tfrecord_path)

    processed_fname = tfrecord_path + "processed_" + fname.split("/")[-1]
    dataset_size = 0
    print("Generating the file of {} ...".format(processed_fname))

    with open(processed_fname, 'w') as fw:
        with open(fname, 'rt') as fr:
            for line in fr:
                line = line.strip()
                fields = line.split('\t')
                
                c_id = fields[0]
                context = fields[1]

                if fields[2] != "NA":
                    personas = [persona for persona in fields[2].split('|')]
                    for p_id, persona in enumerate(personas):
                        p_id = str(100000 + int(c_id)*10 + 0)
                        dataset_size += 1
                        fw.write("\t".join([c_id, context, p_id, persona, 'follow']))
                        fw.write('\n')

                if fields[3] != "NA":
                    personas = [persona for persona in fields[3].split('|')]
                    for p_id, persona in enumerate(personas):
                        p_id = str(100000 + int(c_id)*10 + p_id + 1)
                        dataset_size += 1
                        fw.write("\t".join([c_id, context, p_id, persona, 'unfollow']))
                        fw.write('\n')
    
    print("{} dataset_size: {}".format(processed_fname, dataset_size))            
    return processed_fname


class InputExample(object):
    def __init__(self, guid, text_a_id, text_a, text_b_id, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a_id = text_a_id
        self.text_a = text_a
        self.text_b_id = text_b_id
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, text_a_id, text_b_id, input_ids, input_mask, segment_ids, label_id):
        self.text_a_id = text_a_id
        self.text_b_id = text_b_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def read_processed_file(input_file):
    lines = []
    num_lines = sum(1 for line in open(input_file, 'r'))
    with open(input_file, 'r') as f:
        for line in tqdm(f, total=num_lines):
            concat = []
            temp = line.rstrip().split('\t')
            concat.append(temp[0]) # context id
            concat.append(temp[1]) # context
            concat.append(temp[2]) # persona id
            concat.append(temp[3]) # persona
            concat.append(temp[4]) # label
            lines.append(concat)
    return lines


def create_examples(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, str(i))
        text_a_id = line[0]
        text_a = tokenization.convert_to_unicode(line[1])
        text_b_id = line[2]
        text_b = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[-1])
        examples.append(InputExample(guid=guid, text_a_id=text_a_id, text_a=text_a, text_b_id=text_b_id, text_b=text_b, label=label))
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}  # label
    for (i, label) in enumerate(label_list):  # ['0', '1']
        label_map[label] = i

    features = []  # feature
    for (ex_index, example) in enumerate(examples):
        text_a_id = int(example.text_a_id)
        text_b_id = int(example.text_b_id)

        text_a_sentences = example.text_a.split(" _eos_ ")
        text_b_sentences = example.text_b.split(" _eos_ ")
        text_a_sentences = text_a_sentences[-max_sentence_a_num:]
        text_b_sentences = text_b_sentences[-max_sentence_b_num:]
        while len(text_a_sentences) < max_sentence_a_num:
            text_a_sentences.append("sentence a padding")
        while len(text_b_sentences) < max_sentence_b_num:
            text_b_sentences.append("sentence b padding")
        assert len(text_a_sentences) == max_sentence_a_num
        assert len(text_b_sentences) == max_sentence_b_num

        input_ids = []
        input_mask = []
        segment_ids = []

        for text_a_sentence in text_a_sentences:
            for text_b_sentence in text_b_sentences:
                text_a_sentence_tokens = tokenizer.tokenize(text_a_sentence)
                text_a_sentence_tokens = text_a_sentence_tokens[:max_sentence_a_len]

                text_b_sentence_tokens = tokenizer.tokenize(text_b_sentence)
                text_b_sentence_tokens = text_b_sentence_tokens[:max_sentence_b_len]

                _truncate_seq_pair(text_a_sentence_tokens, text_b_sentence_tokens, max_sentence_a_len + max_sentence_b_len - 3)

                tokens_temp = []
                segment_ids_temp = []

                tokens_temp.append("[CLS]")
                segment_ids_temp.append(0)
                for token in text_a_sentence_tokens:
                    tokens_temp.append(token)
                    segment_ids_temp.append(0)
                tokens_temp.append("[SEP]")
                segment_ids_temp.append(0)

                for token in text_b_sentence_tokens:
                    tokens_temp.append(token)
                    segment_ids_temp.append(1)
                tokens_temp.append("[SEP]")
                segment_ids_temp.append(1)

                input_ids_temp = tokenizer.convert_tokens_to_ids(tokens_temp)

                input_mask_temp = [1] * len(input_ids_temp)  # mask

                # Zero-pad up to the sequence length.
                while len(input_ids_temp) < (max_sentence_a_len + max_sentence_b_len):
                    input_ids_temp.append(0)
                    input_mask_temp.append(0)
                    segment_ids_temp.append(0)

                assert len(input_ids_temp) == (max_sentence_a_len + max_sentence_b_len)
                assert len(input_mask_temp) == (max_sentence_a_len + max_sentence_b_len)
                assert len(segment_ids_temp) == (max_sentence_a_len + max_sentence_b_len)

                input_ids.extend(input_ids_temp)
                input_mask.extend(input_mask_temp)
                segment_ids.extend(segment_ids_temp)

        label_id = label_map[example.label]

        # len(input_ids) = (max_sentence_a_len + max_sentence_b_len) * max_sentence_a_num * max_sentence_b_num

        if ex_index%2000 == 0:
            print('convert_{}_examples_to_features'.format(ex_index))

        features.append(
            InputFeatures(  # object
                text_a_id=text_a_id,
                text_b_id=text_b_id,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id))

    return features


def write_instance_to_example_files(instances, output_files):
    writers = []

    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        features = collections.OrderedDict()
        features["text_a_id"] = create_int_feature([instance.text_a_id])
        features["text_b_id"] = create_int_feature([instance.text_b_id])
        features["input_ids"] = create_int_feature(instance.input_ids)
        features["input_mask"] = create_int_feature(instance.input_mask)
        features["segment_ids"] = create_int_feature(instance.segment_ids)
        features["label_ids"] = create_float_feature([instance.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        # if inst_index < 5:
        # 	print("*** Example ***")
        # 	print("text_a_id: %s" % instance.text_a_id)
        # 	print("text_b_id: %s"  % instance.text_b_id)
        # 	print("input_ids: %s" % " ".join([str(tokenization.printable_text(x)) for x in instance.input_ids]))
        # 	print("input_mask: %s" % " ".join([str(tokenization.printable_text(x)) for x in instance.input_mask]))
        # 	print("segment_ids: %s" % " ".join([str(tokenization.printable_text(x)) for x in instance.segment_ids]))
        # 	print("label_id: %s" % instance.label_id)

    print("write_{}_instance_to_example_files".format(total_written))

    for feature_name in features.keys():
        feature = features[feature_name]
        values = []
    if feature.int64_list.value:
        values = feature.int64_list.value
    elif feature.float_list.value:
        values = feature.float_list.value
    tf.logging.info(
        "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()


def create_int_feature(values):
	feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
	return feature

def create_float_feature(values):
	feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
	return feature



if __name__ == "__main__":

    FLAGS = tf.flags.FLAGS

    print_configuration_op(FLAGS)

    train_filename = load_dataset(FLAGS.train_file)
    valid_filename = load_dataset(FLAGS.valid_file)
    test_filename = load_dataset(FLAGS.test_file)

    filenames = [train_filename, valid_filename, test_filename]
    filetypes = ["train", "valid", "test"]
    files = zip(filenames, filetypes)

    label_list = ["unfollow", "follow"]
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    # [exp1, exp2...] example X is a class object: {guid, text_a(str), text_b(str), label(str)}
    for (filename, filetype) in files:
        examples = create_examples(read_processed_file(filename), filetype)
        features = convert_examples_to_features(examples, label_list, FLAGS.max_seq_length, tokenizer)
        new_filename = filename[:-4] + ".tfrecord"
        write_instance_to_example_files(features, [new_filename])
        print('Convert {} to {} done'.format(filename, new_filename))

    print("Sub-process(es) done.")
