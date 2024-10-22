CUDA_VISIBLE_DEVICES=0 python run_pretrain_imagine_gpt2.py \
--model_type gpt2 \
--model_name_or_path "openai-community/gpt2-large" \
--task_name atomic \
--output_dir ./output/gpt2-large \
--train_file ../../train_syntheticVQA.jsonl \
--dev_file ../../dev_syntheticVQA.jsonl \
--max_seq_length 128 \
--do_train --do_eval \
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 8 \
--learning_rate 1e-5 \
--num_train_epochs 2 \
--warmup_proportion 0.05 \
--evaluate_during_training \
--per_gpu_eval_batch_size 4  \
--save_steps 10000 \
--margin 1.0 \
--overwrite_output_dir