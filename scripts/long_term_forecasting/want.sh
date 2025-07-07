export CUDA_VISIBLE_DEVICES=0

seq_len=84 # Increase sequence length for long-term forecasting = 1 week
model=CALF

# for pred_len in 96 192 336 720
for pred_len in 12 24 48 168 
do

python3 run.py \
    --root_path ./datasets/network/processed \
    --data_path snmp_first10.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id traffic_$model'_'$seq_len'_'$pred_len \
    --data custom \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 8 \
    --learning_rate 0.0005 \
    --train_epochs 20 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --gpt_layers 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 5 \
    --task_loss smooth_l1 \
    --feature_loss smooth_l1 \
    --output_loss smooth_l1 \
    --enc_in 10 \
    --c_out 10 \
    --target SACR_SUNN_in \

echo '====================================================================================================================='
done
