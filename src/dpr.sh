WANDB_PROJECT=dpr_retrieval WANDB_LOG_MODEL=false \
    python -m torch.distributed.launch --nproc_per_node=1 -m tevatron.driver.train \
      --output_dir output \
      --model_name_or_path cointegrated/LaBSE-en-ru \
      --untie_encoder \
      --save_steps 200 \
      --save_total_limit 3 \
      --dataset_name d0rj/dpr_data_draft \
      --per_device_train_batch_size 13 \
      --gradient_accumulation_steps 10 \
      --positive_passage_no_shuffle \
      --train_n_passages 2 \
      --learning_rate 1e-5 \
      --q_max_len 512 \
      --p_max_len 512 \
      --num_train_epochs 3 \
      --logging_steps 100 \
      --negatives_x_device \
      --dataset_proc_num 6 \
      --overwrite_output_dir
