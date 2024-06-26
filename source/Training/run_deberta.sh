CUDA_VISIBLE_DEVICES=0 python run_pretrain_imagine_encoder.py \
--model_type deberta-mlm \
--model_name_or_path "microsoft/deberta-v3-large" \
--task_name atomic \
--output_dir ./output/deberta-v3-large \
--train_file ../../train_syntheticVQA.jsonl \
--dev_file ../../dev_syntheticVQA.jsonl \
--max_seq_length 128 \
--do_train --do_eval \
--per_gpu_train_batch_size 2 \
--gradient_accumulation_steps 16 \
--learning_rate 7e-6 \
--num_train_epochs 2 \
--warmup_proportion 0.05 \
--evaluate_during_training \
--per_gpu_eval_batch_size 2  \
--save_steps 2000 \
--margin 1.0 \
--overwrite_output_dir \
--max_words_to_mask 4 \