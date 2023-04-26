# This is the script to run prompt-based fine-tuning to generate the pesudo labels for train and dev set.
TASK_TYPE=glue
MODEL_TYPE=prompting
MAX_LENGTH=256
CHECKPOINT=roberta-large
lr=2e-5
for TASK_NAME in SNLI QNLI RTE STS-B; do
    for seed in 13; do
        CUDA_VISIBLE_DEVICES=0 python run_prompt_ft.py \
            --task_type ${TASK_TYPE} \
            --model_type ${MODEL_TYPE} \
            --downstream_task_name ${TASK_NAME} \
            --train_file data/k-shot/${TASK_NAME}/16-${seed} \
            --validation_file data/k-shot/${TASK_NAME}/16-${seed} \
            --test_file data/glue_pretrain/${TASK_NAME} \
            --model_name_or_path ${CHECKPOINT} \
            --do_train \
            --do_eval \
            --do_predict \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --max_seq_length ${MAX_LENGTH} \
            --save_strategy steps \
            --evaluation_strategy steps \
            --max_steps 1000 \
            --eval_steps 100 \
            --save_steps 100 \
            --learning_rate ${lr} \
            --weight_decay 0.01 \
            --warmup_ratio 0.06 \
            --load_best_model_at_end \
            --save_total_limit 1 \
            --run_pseduo_label \
            --output_dir saved_pesudo_labels_glue/${MODEL_TYPE}_${TASK_NAME}_${seed}_${lr};
    done;
done