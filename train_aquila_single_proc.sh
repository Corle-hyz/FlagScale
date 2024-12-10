export PYTHONPATH=/share/project/heyongzhe/fix_node_rank_bug/FlagScale/megatron:/share/project/heyongzhe/fix_node_rank_bug/FlagScale
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=8
export RZDV_ENDPOINT=localhost:37832
export RZDV_BACKEND=c10d
export MASTER_ADDR=localhost

# --enable-hetero
# --enable-simulator
# --distributed-backend dummy

python ./flagscale/train/train_aquila.py --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --disable-bias-linear --use-flash-attn --sequence-parallel --use-distributed-optimizer --use-mcore-models --transformer-impl transformer_engine --enable-hetero --enable-simulator --distributed-backend dummy --hetero-process-meshes 1 1 1 8 1 --hetero-device-types A800 --hetero-current-device-type A800 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --recompute-granularity-per-stage-micro-batch '[1, 1, 1]' --recompute-method-per-stage-micro-batch '[1, 1, 1]' --recompute-num-layers-per-stage-micro-batch '[1, 1, 1]' --bf16 --attention-softmax-in-fp32 --accumulate-allreduce-grads-in-fp32 --log-interval 1 --log-throughput --tensorboard-log-interval 1 --wandb-project aquila2 --wandb-exp-name test --tensorboard-dir /share/project/heyongzhe/fix_node_rank_bug/FlagScale/outputs/tensorboard --wandb-save-dir /share/project/heyongzhe/fix_node_rank_bug/FlagScale/outputs/wandb --num-layers 4 --hidden-size 4096 --num-attention-heads 32 --seq-length 2048 --max-position-embeddings 2048 --norm-epsilon 1e-05 --use-rotary-position-embeddings --no-position-embedding --swiglu --multiple-of 256 --normalization RMSNorm --rotary-interleaved-patch --untie-embeddings-and-output-weights --init-method-std 0.0165 --attention-dropout 0.0 --hidden-dropout 0.0 --weight-decay 0.1 --clip-grad 1.0 --train-samples 128 --global-batch-size 8 --micro-batch-size 1 --seed 42 --lr 0.0002 --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.95 --lr 0.00015 --min-lr 1.5e-05 --lr-warmup-samples 0 --lr-decay-style cosine --data-path /share/project/caozhou/adaptive_flash_ckpt/FlagScale/data/pile_wikipedia_demo --split 1 --tokenizer-type AquilaTokenizerFS --vocab-file ./examples/aquila/tokenizer/vocab.json --merge-file ./examples/aquila/tokenizer/merges.txt --special-tokens-file ./examples/aquila/tokenizer/special_tokens.txt --vocab-size 100008
