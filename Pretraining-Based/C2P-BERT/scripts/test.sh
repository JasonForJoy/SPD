
CUDA_VISIBLE_DEVICES=0 python -u ../test.py \
    --test_dir ../data_tfrecord/processed_test_both_revised_cand_10.tfrecord \
    --vocab_file ../../uncased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file ../../uncased_L-12_H-768_A-12/bert_config.json \
    --max_seq_length 200 \
    --eval_batch_size 50 \
    --restore_model_dir ../output/1631501715 > log_test_BERT_cand_10.txt 2>&1 &
