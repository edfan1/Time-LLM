model_name=TimeLLM
train_epochs=100
learning_rate=0.01
llama_layers=32

# master_port=00097
num_process=3
batch_size=24
d_model=32
d_ff=128

comment='TimeLLM-snmp2018'

#--multi_gpu  --main_process_port $master_port

# accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --num_machines 3 --machine_rank 0 run_main.py \
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path snmp_2018_1hourinterval.csv \
  --model_id snmp2018_512_96 \
  --model $model_name \
  --data snmp2018 \
  --features M \
  --target 'SACR_SUNN_in' \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

# accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path snmp_2018_1hourinterval.csv \
#   --model_id snmp2018_512_192 \
#   --model $model_name \
#   --data snmp2018 \
#   --features M \
#   --seq_len 512 \
#   --label_len 48 \
#   --pred_len 192 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 32 \
#   --d_ff 128 \
#   --batch_size $batch_size \
#   --learning_rate 0.02 \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment

# accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path snmp_2018_1hourinterval.csv \
#   --model_id snmp2018_512_336 \
#   --model $model_name \
#   --data snmp2018 \
#   --features M \
#   --seq_len 512 \
#   --label_len 48 \
#   --pred_len 336 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --lradj 'COS'\
#   --learning_rate 0.001 \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment

# accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path snmp_2018_1hourinterval.csv \
#   --model_id snmp2018_512_720 \
#   --model $model_name \
#   --data snmp2018 \
#   --features M \
#   --seq_len 512 \
#   --label_len 48 \
#   --pred_len 720 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment