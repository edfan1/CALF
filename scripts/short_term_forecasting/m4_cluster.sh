export CUDA_VISIBLE_DEVICES=0

model_name=CALF

# Loop through different seasonal patterns
for pattern in 'Monthly' 'Quarterly' 'Yearly'
do
  # Loop through clustering methods
  for method in 'Pearson' 'Spearman' 'Feature'
  do
    # Get number of clusters for this method and pattern
    cluster_count=$(ls -1 ./datasets/m4_clustered/${pattern}/${method}/ | wc -l)
    
    # Loop through all clusters
    for cluster_id in $(seq 1 $cluster_count)
    do
      # Create descriptive model ID
      model_id="m4_${pattern}_${method}_cluster${cluster_id}"
      log_file="logs/${model_id}.log"
      
      echo "Running experiment: ${model_id}"
      echo "Logging to: ${log_file}"
      
      {
        python -u run.py \
          --task_name short_term_forecast \
          --is_training 1 \
          --root_path ./datasets/m4_clustered \
          --seasonal_patterns ${pattern} \
          --model_id ${model_id} \
          --model ${model_name} \
          --data m4_clustered \
          --cluster_method ${method} \
          --cluster_id ${cluster_id} \
          --features M \
          --e_layers 2 \
          --d_layers 1 \
          --factor 3 \
          --enc_in 1 \
          --dec_in 1 \
          --c_out 1 \
          --train_epochs 200 \
          --batch_size 512 \
          --d_model 768 \
          --d_ff 768 \
          --n_heads 4 \
          --des 'Exp' \
          --itr 1 \
          --learning_rate 0.001 \
          --r 8 \
          --lora_alpha 32 \
          --lora_dropout 0.1 \
          --patience 20 \
          --gpt_layers 6 \
          --task_loss smape \
          --output_loss mase \
          --feature_loss smooth_l1
      } > ${log_file} 2>&1
      
      echo "Finished experiment: ${model_id}"
      echo "====================================================="
    done
  done
done