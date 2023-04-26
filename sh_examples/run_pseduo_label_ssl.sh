# This is the script to run prompt-based fine-tuning to generate the pesudo labels for train and dev set.
TASK_TYPE=ssl
MODEL_TYPE=dart
CHECKPOINT=roberta-large
MAX_LENGTH=256
NUMBER_LABELS=200
for lr in 1e-5; do
    for TASK_NAME in ag_news; do
        for seed in 1; do
            CUDA_VISIBLE_DEVICES=0 python run_prompt_ft.py \
                --task_type ${TASK_TYPE} \
                --model_type ${MODEL_TYPE} \
                --downstream_task_name ${TASK_NAME} \
                --seed ${seed} \
                --num_labelled_data ${NUMBER_LABELS} \
                --train_file data/${TASK_NAME} \
                --validation_file data/${TASK_NAME} \
                --test_file data/${TASK_NAME} \
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
                --output_dir saved_pesudo_labels_${TASK_TYPE}/${MODEL_TYPE}_${TASK_NAME}_${seed}_${lr}_${NUMBER_LABELS};
        done;
    done;
done