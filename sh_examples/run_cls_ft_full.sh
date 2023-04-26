# This is the script to run cls-based fine-tuning on the full training set.
TYPE=large
TASK_TYPE=glue
MAX_LENGTH=256
for TASK_NAME in MRPC QQP RTE; do
    CHECKPOINT=roberta-large
    for lr in 1e-5 2e-5 5e-5; do
        CUDA_VISIBLE_DEVICES=0 python run_cls_ft.py \
            --task_type ${TASK_TYPE} \
            --task_name ${TASK_NAME} \
            --train_file data/original/${TASK_NAME} \
            --validation_file data/k-shot/${TASK_NAME}/16-13 \
            --test_file data/k-shot/${TASK_NAME}/16-13 \
            --model_name_or_path ${CHECKPOINT} \
            --do_train \
            --do_eval \
            --do_predict \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
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
            --output_dir saved_${TASK_TYPE}_cls_full/${TYPE}_${TASK_NAME}_${lr};
    done;
done