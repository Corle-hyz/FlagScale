import os
import subprocess
import re
from megatron.training import get_args


def compute_pipeline_parallelism_cost(
        scheme: str='1F1B',
        # num_stages: int=1,
        num_micro_batches: int=1,
        process_mesh: list=None,
        pp_layers_split: list=None,
        fwd_time_per_stage_chunk: list=None,
        bwd_time_per_stage_chunk: list=None,
        comm_time_between_stages: list=None,
        # TODO: add fine-greaied recomputation
):
    # process_mesh: [tp0,cp0,ep0,dp0,pp0,(tp1,cp1,...)]
    # comm_time_between_stages[i] means the comm time between stage i-1 and stage i
    num_pp_stages = sum(process_mesh[3::5])
    assert len(pp_layers_split) ==  num_pp_stages, \
        "\flength of list {num_layers_per_stage} should match {num_stages}"
    assert len(fwd_time_per_stage_chunk) ==  num_pp_stages, \
        "\flength of list {fwd_time_per_stage_chunk} should match {num_stages}"
    assert len(bwd_time_per_stage_chunk) ==  num_pp_stages, \
        "\flength of list {bwd_time_per_stage_chunk} should match {num_stages}"
    assert len(comm_time_between_stages) ==  num_pp_stages, \
        "\flength of list {comm_time_between_stages} should match {num_stages}"
    
    pp_last_stage_time = num_micro_batches * (fwd_time_per_stage_chunk[num_pp_stages-1] + bwd_time_per_stage_chunk[num_pp_stages-1])
    if num_pp_stages==1:
        return num_micro_batches * (fwd_time_per_stage_chunk[num_pp_stages-1] + bwd_time_per_stage_chunk[num_pp_stages-1])
    
    pipeline_cost = 0
    # TODO: consider when comm time > comp time
    # each stage onlt depends on its next stage
    if scheme == '1F1B' or scheme== 'AFAB':
        pipeline_cost = pp_last_stage_time
        for stage_from_last in range(2, num_pp_stages):
            pp_this_stage_overlapped_time = (num_micro_batches-1) * (fwd_time_per_stage_chunk[num_pp_stages-1] + bwd_time_per_stage_chunk[num_pp_stages-1])
            pp_this_stage_compute_time = fwd_time_per_stage_chunk[num_pp_stages-stage_from_last] + bwd_time_per_stage_chunk[num_pp_stages-stage_from_last]
            pp_last_stage_overall_time = pipeline_cost + 2 * comm_time_between_stages[num_pp_stages-stage_from_last+1]
            # not consider the situation that comm stucks the comp
            # which means the comm time should no more than the comp time(fwd time)
            pipeline_cost = pp_this_stage_compute_time + max(pp_last_stage_overall_time, pp_this_stage_overlapped_time)
    else:
        raise(ValueError("Scheme must be '1F1B' or 'AFAB'."))

    return pipeline_cost


def simulator(
        process_mesh: list=None,
        num_layers: int=None,
        simulated_rank: int=None
):
    args = get_args()
    
    os.environ["PYTHONPATH"] = "/share/project/heyongzhe/fix_node_rank_bug/FlagScale/megatron:/share/project/heyongzhe/fix_node_rank_bug/FlagScale"
    os.environ["CUDA_VISIBLE_DEVICES"] = 0
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = 1
    os.environ["RANK"] = simulated_rank
    os.environ["LOCAL_RANK"] = simulated_rank
    os.environ["WORLD_SIZE"] = args.world_size
    os.environ["RZDV_ENDPOINT"]="localhost:37832"
    os.environ["RZDV_BACKEND"]="c10d"
    os.environ["MASTER_ADDR"]="localhost"

    simulation_arguments = " --enable-hetero --enable-simulator --distributed-backend dummy"
    train_command = "python ./flagscale/train/train_aquila.py --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --disable-bias-linear --use-flash-attn --sequence-parallel --use-distributed-optimizer --use-mcore-models --transformer-impl transformer_engine --hetero-process-meshes 1 1 1 8 1 --hetero-device-types A800 --hetero-current-device-type A800 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --recompute-granularity-per-stage-micro-batch '[1, 1, 1]' --recompute-method-per-stage-micro-batch '[1, 1, 1]' --recompute-num-layers-per-stage-micro-batch '[1, 1, 1]' --bf16 --attention-softmax-in-fp32 --accumulate-allreduce-grads-in-fp32 --log-interval 1 --log-throughput --tensorboard-log-interval 1 --wandb-project aquila2 --wandb-exp-name test --tensorboard-dir /share/project/heyongzhe/fix_node_rank_bug/FlagScale/outputs/tensorboard --wandb-save-dir /share/project/heyongzhe/fix_node_rank_bug/FlagScale/outputs/wandb --num-layers 4 --hidden-size 4096 --num-attention-heads 32 --seq-length 2048 --max-position-embeddings 2048 --norm-epsilon 1e-05 --use-rotary-position-embeddings --no-position-embedding --swiglu --multiple-of 256 --normalization RMSNorm --rotary-interleaved-patch --untie-embeddings-and-output-weights --init-method-std 0.0165 --attention-dropout 0.0 --hidden-dropout 0.0 --weight-decay 0.1 --clip-grad 1.0 --train-samples 128 --global-batch-size 8 --micro-batch-size 1 --seed 42 --lr 0.0002 --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.95 --lr 0.00015 --min-lr 1.5e-05 --lr-warmup-samples 0 --lr-decay-style cosine --data-path /share/project/caozhou/adaptive_flash_ckpt/FlagScale/data/pile_wikipedia_demo --split 1 --tokenizer-type AquilaTokenizerFS --vocab-file ./examples/aquila/tokenizer/vocab.json --merge-file ./examples/aquila/tokenizer/merges.txt --special-tokens-file ./examples/aquila/tokenizer/special_tokens.txt --vocab-size 100008"

    # os.system(train_command+simulation_arguments)
    result = subprocess.run(train_command+simulation_arguments, capture_output=True, text=True)
    output = result.stdout.strip()
    # example output: "[simulatior output] forward: 12.34, backward: 56.78, communication: 90.12"
    match = re.search(r"forward:\s*([\d.]+),\s*backward:\s*([\d.]+),\s*communication:\s*([\d.]+)", output)

    if match:
        fwd_time = float(match.group(1))
        bwd_time = float(match.group(2))
        comm_time = float(match.group(3))
        print("forward:", fwd_time)
        print("backward:", bwd_time)
        print("communication:", comm_time)
    else:
        raise(ValueError("Results not found. Example output: \"[simulatior output] forward: 12.34, backward: 56.78, communication: 90.12\""))
    return fwd_time, bwd_time, comm_time


# call simulator to obtain the execution of each stage
def simulate_pipeline_parallelism_per_stage_time(
        process_mesh: list=None,
        pp_layers_split: list=None,
        fwd_time_per_stage_chunk: list=None,
        bwd_time_per_stage_chunk: list=None,
        comm_time_between_stages: list=None,
):

    for stage,num_layers in enumerate(pp_layers_split):
        # TODO: confirm simulated_rank for different stage
        simulated_rank = 0
        fwd_time, bwd_time, comm_time = simulator(process_mesh, stage, num_layers, simulated_rank)
        fwd_time_per_stage_chunk.append(fwd_time)
        bwd_time_per_stage_chunk.append(bwd_time)
        comm_time_between_stages.append(comm_time)



def analyze_pp_time(
        scheme: str='1F1B',
        num_micro_batches: int=1,
        process_mesh: list=None,
        pp_layers_split: list=None
    ):
    fwd_time_per_stage_chunk = []
    bwd_time_per_stage_chunk = []
    comm_time_between_stages = []

    simulate_pipeline_parallelism_per_stage_time(
        process_mesh=process_mesh,
        pp_layers_split=pp_layers_split,
        fwd_time_per_stage_chunk=fwd_time_per_stage_chunk,
        bwd_time_per_stage_chunk=bwd_time_per_stage_chunk,
        comm_time_between_stages=comm_time_between_stages
    )

    pipeline_cost = compute_pipeline_parallelism_cost(
        scheme=scheme,
        num_micro_batches=num_micro_batches,
        process_mesh=process_mesh,
        pp_layers_split=pp_layers_split,
        fwd_time_per_stage_chunk=fwd_time_per_stage_chunk,
        bwd_time_per_stage_chunk=bwd_time_per_stage_chunk,
        comm_time_between_stages=comm_time_between_stages
    )

    return pipeline_cost
