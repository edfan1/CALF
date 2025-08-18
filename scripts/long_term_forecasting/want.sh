export CUDA_VISIBLE_DEVICES=0

seq_len=168
model=CALF

# Loop through different datasets
for dataset in snmp_2018_1hourinterval_Feature_group_1.csv snmp_2018_1hourinterval_Feature_group_2.csv snmp_2018_1hourinterval_Feature_group_3.csv
do
  # Extract dataset name without extension
  dataset_name=$(basename "${dataset%.*}")
  # Replace [[ ]] syntax with [ ] syntax
  if [ -n "$(echo "$dataset_name" | grep "group_1")" ]; then
    enc_in=46
    c_out=46
    echo "Setting group 1 dimensions: enc_in=$enc_in, c_out=$c_out"
  elif [ -n "$(echo "$dataset_name" | grep "group_2")" ]; then
    enc_in=44
    c_out=44
    echo "Setting group 2 dimensions: enc_in=$enc_in, c_out=$c_out"
  elif [ -n "$(echo "$dataset_name" | grep "group_3")" ]; then
    enc_in=3
    c_out=3
    echo "Setting group 3 dimensions: enc_in=$enc_in, c_out=$c_out"
  elif [ -n "$(echo "$dataset_name" | grep "group_4")" ]; then
    enc_in=3
    c_out=3
    echo "Setting group 4 dimensions: enc_in=$enc_in, c_out=$c_out"
  else
    # Default fallback
    enc_in=96
    c_out=96
    echo "WARNING: Unknown dataset type. Using default dimensions: enc_in=$enc_in, c_out=$c_out"
  fi
  
  # Loop through different prediction lengths
  for pred_len in 12
  do
    # Create log filename based on dataset and prediction length
    log_file="logs/${dataset_name}_${model}_${seq_len}_${pred_len}.log"
    echo "Running experiment with dataset: $dataset_name, pred_len: $pred_len"
    echo "Logging output to: $log_file"
    
    # Run the command and redirect output to log file
    {
      echo "=== EXPERIMENT START: $(date) ==="
      echo "Dataset: $dataset_name, Model: $model, Seq_len: $seq_len, Pred_len: $pred_len"
      echo "Model dimensions: enc_in=$enc_in, c_out=$c_out"
      
      python3 run.py \
          --root_path ./datasets/network/clustering_snmp_2018_1hourinterval/grouped_columns \
          --data_path $dataset \
          --is_training 1 \
          --task_name long_term_forecast \
          --model_id ${dataset_name}_${model}'_'${seq_len}'_'${pred_len} \
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
          --enc_in $enc_in \
          --c_out $c_out \
      
      echo "=== EXPERIMENT END: $(date) ==="
    } > "$log_file" 2>&1
    
    echo "Finished experiment. Log saved to: $log_file"
    echo '====================================================================================================================='
  done
done
