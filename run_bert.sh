LAMBDA=0.1
SEED=66
SAVE_DIR=CRAG_lambda${LAMBDA}

python ../run.py --do_train \
  --data_dir ./dataset/docred \
  --transformer_type bert \
  --model_name_or_path bert-base-cased \
  --save_path ${SAVE_DIR} \
  --train_file train_annotated.json \
  --dev_file dev.json \
  --train_batch_size 4 \
  --test_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_labels 4 \
  --lr_transformer 5e-5 \
  --max_grad_norm 1.0 \
  --evi_thresh 0.2 \
  --evi_lambda ${LAMBDA} \
  --warmup_ratio 0.06 \
  --num_train_epochs 30.0 \
  --seed ${SEED} \
  --num_class 97
