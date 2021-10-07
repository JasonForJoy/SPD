
train_file=../../../data_PMPC/train_both_revised.txt
valid_file=../../../data_PMPC/valid_both_revised_cand_10.txt
vocab_file=../../../data_PMPC/vocab.txt
char_vocab_file=../../../data_PMPC/char_vocab.txt
embedded_vector_file=../../../data_PMPC/filtered.glove.42B.300d.txt

max_context_len=150
max_persona_len=50
max_word_length=18
num_layer=1
embedding_dim=300
rnn_size=200

batch_size=128
lambda=0
dropout_keep_prob=0.8
num_epochs=1000
evaluate_every=100

CUDA_VISIBLE_DEVICES=0 python -u ../train.py \
                --train_file $train_file \
                --valid_file $valid_file \
                --vocab_file $vocab_file \
                --char_vocab_file $char_vocab_file \
                --embedded_vector_file $embedded_vector_file \
                --max_context_len $max_context_len \
                --max_persona_len $max_persona_len \
                --max_word_length $max_word_length \
                --num_layer $num_layer \
                --embedding_dim $embedding_dim \
                --rnn_size $rnn_size \
                --batch_size $batch_size \
                --l2_reg_lambda $lambda \
                --dropout_keep_prob $dropout_keep_prob \
                --num_epochs $num_epochs \
                --evaluate_every $evaluate_every > log_train_BOW_cand_10.txt 2>&1 &
