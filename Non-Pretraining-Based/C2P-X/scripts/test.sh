
latest_checkpoint=../output/1631263935/checkpoints
echo $latest_checkpoint

test_file=../../../data_PMPC/test_both_revised_cand_10.txt
vocab_file=../../../data_PMPC/vocab.txt
char_vocab_file=../../../data_PMPC/char_vocab.txt
output_file=${latest_checkpoint}/output_test.txt

max_context_len=150
max_persona_len=50
max_word_length=18
batch_size=128

CUDA_VISIBLE_DEVICES=0 python -u ../test.py \
                  --test_file $test_file \
                  --vocab_file $vocab_file \
                  --char_vocab_file $char_vocab_file \
                  --output_file $output_file \
                  --max_context_len $max_context_len \
                  --max_persona_len $max_persona_len \
                  --max_word_length $max_word_length \
                  --batch_size $batch_size \
                  --checkpoint_dir $latest_checkpoint > log_test_BOW_cand_10.txt 2>&1 &
