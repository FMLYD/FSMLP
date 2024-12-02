if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2024

seq_len=96

for learning_rate in 0.01  
do
for fc_dropout in     0.7
do
for d_ff in  256     
do
for d_model in    256   
do
for m_layers in     1 
do
for m_model in  1
do
for e_layers in   3 
do
for f_model in 1
do
for dropout in     0.3
do
for model_name in  Log
# DSCNN DSCNN_fft_dct DSCNN_fft 
do
for pred_len in  96 
do
  python -u run_longExp.py \
    --random_seed 2021 \
    --is_training 1 \
    --root_path ./dataset/\
    --data_path ETTh1.csv\
    --model_id ETTh1'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --e_layers $e_layers\
    --gpu 1\
    --num_workers 0\
    --n_heads 1\
    --d_model $d_model \
    --d_ff $d_ff \
    --dropout $dropout\
    --fc_dropout $fc_dropout\
    --period 96  \
    --m_layer $m_layers\
    --m_model $m_model\
    --patch_len 1  \
    --stride 1  \
    --f_model $f_model\
    --des Exp \
    --pct_start 0.2 \
    --train_epochs 100 \
    --patience 10\
    --itr 1 --batch_size 256  --learning_rate $learning_rate|tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$learning_rate.log
done
done
done
done 
done
done
done
done
done
done
done