DEVICE=0;
ssl_type=wavlm-large
pool_type=AttentiveStatisticsPooling
model_name=EmoSphere-SER

for seed in 7; do
    CUDA_VISIBLE_DEVICES=$DEVICE python train_eval_files/train_emosphereser.py \
        --seed=${seed} \
        --ssl_type=${ssl_type} \
        --batch_size=64 \
        --accumulation_steps=4 \
        --lr=1e-5 \
        --epochs=20 \
        --model_path=${model_name} || exit 0;

    # test
    CUDA_VISIBLE_DEVICES=$DEVICE python train_eval_files/test_emosphereser.py \
        --ssl_type=${ssl_type} \
        --model_path=${model_name}  \
        --store_path=${model_name}.txt || exit 0;
    
    # eval
    CUDA_VISIBLE_DEVICES=$DEVICE python train_eval_files/eval_emosphereser.py \
        --ssl_type=${ssl_type} \
        --model_path=${model_name}  \
        --store_path=${model_name}.txt || exit 0;
    
done
