
latest_checkpoint=../output/1631263935/checkpoints
echo $latest_checkpoint

test_file=../../../data_PMPC/test_both_revised_cand_10.txt
vocab_file=../../../data_PMPC/vocab.txt
char_vocab_file=../../../data_PMPC/char_vocab.txt
output_file=${latest_checkpoint}/output_test.txt

max_utter_num=8
max_utter_len=20
max_profile_num=5
max_profile_len=15
max_word_length=18
batch_size=128

CUDA_VISIBLE_DEVICES=0 python -u ../test.py \
                  --test_file $test_file \
                  --vocab_file $vocab_file \
                  --char_vocab_file $char_vocab_file \
                  --output_file $output_file \
                  --max_utter_num $max_utter_num \
                  --max_utter_len $max_utter_len \
                  --max_profile_num $max_profile_num \
                  --max_profile_len $max_profile_len \
                  --max_word_length $max_word_length \
                  --batch_size $batch_size \
                  --checkpoint_dir $latest_checkpoint > log_test_BOW_can_10.txt 2>&1 &
