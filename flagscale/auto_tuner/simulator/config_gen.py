from itertools import product
from itertools import combinations


class DevicesInfo:
    def __init__(self, device_type_list: list, device_num_list: list):
        assert len(device_type_list) == len(device_num_list), \
            "\flength of list {device_type_list} should match {device_num_list}"
        self.device_type_list = device_type_list
        self.device_num_list = device_num_list
        self.device_types_count = len(device_type_list)
        self.possible_parallelisms = []

class HeteroConfig:
    def __init__(self, 
                 mesh, 
                 device_types, 
                 pp_layer_split, 
                 recompute_granularity = None, 
                 recompute_method = None,
                 recompute_num_layers = None):
        self.mesh = mesh
        self.device_types = device_types
        self.pp_layer_split = pp_layer_split
        # self.micro_batch_size = 1
        self.recompute_granularity = recompute_granularity
        self.recompute_method = recompute_method
        self.recompute_num_layers = recompute_num_layers

        self.simulated_time = 0.0


def generate_hetero_meshes(
        devices_info: DevicesInfo,
        global_batch_size: int=None,
        num_layers: int=None,
):
    def enumerate_parallelism(
        device_num: int=None
        ):
        possible_parallelisms = []
        for tp in range(1, device_num+1):
            for dp in range(1, device_num//tp+1):
                if device_num % (dp*tp) == 0:
                    pp = device_num // (dp*tp)
                    # mesh: [tp, cp, ep, dp, pp]
                    possible_parallelisms.append([tp, 1, 1, dp, pp])
        return possible_parallelisms
    
    # enumerate all possible meshes for each kind of device
    for i in range(devices_info.device_types_count):
        device_num = devices_info.device_num_list[i]
        devices_info.possible_parallelisms.append(enumerate_parallelism(device_num))
    
    def is_legal_combination(comb: list):
        pp = sum(comb[4::5])
        # check dp is legal
        max_dp = global_batch_size // pp
        for dp in comb[3::5]:
            if max_dp % dp != 0:
                return False
        return True

    def combine_possible_parallelisms(possible_parallelisms):
        ''' example
        devices_info.possible_parallelisms = [
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]],
            [[1, 2, 3, 4, 5]]
        ]
        combine_possible_parallelisms(devices_info.possible_parallelisms)
        -> [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        -> [6, 7, 8, 9, 0, 1, 2, 3, 4, 5]
    }
        '''
        all_combinations = product(*possible_parallelisms)
        results = [sum(comb, []) for comb in all_combinations]
        for result in results:
            if not is_legal_combination(result):
                results.remove(result)
        return results
    
    return combine_possible_parallelisms(devices_info.possible_parallelisms)
    

def split_layers(num_layers, pp_stages):
    results = []
    for split_points in combinations(range(1, num_layers), pp_stages - 1):
        splits = [split_points[0]] + [split_points[i] - split_points[i - 1] for i in range(1, len(split_points))] + [num_layers - split_points[-1]]
        results.append(splits)
    return results


def generate_fine_grained_recompute(
    pp_stages: int,
    num_micro_batches: int,
    pp_layer_split: list
):
    '''
    # pp 3 stages
    # recompute_granularity_per_stage_micro_batch:
      # [stages, num_mbs, enable/disable, ...]
      # - [1, 1, 0, 0, 0]
      # - [1, 30, 1, 2, 1]
      # - [1, 30, 1, 2, 1]
    # recompute_method_per_stage_micro_batch:
      # [stages, num_mbs, bloclk/uniform, ...]
      # - [1, 1, 0, 0, 0]
      # - [1, 30, 0, 2, 0]
      # - [1, 30, 1, 2, 1]
    # recompute_num_layers_per_stage_micro_batch:
      # [stages, num_mbs, num_layers, ...]
      # - [1, 1, 0, 0, 0]
      # - [1, 30, 1, 2, 1]
      # - [1, 30, 2, 2, 2]
    '''
    # to avoid huge search space, we assume the config of each stage is a list of length 5
    # for now we only change the granularity, leave out method and num_layers

    # if the num_micro_batches is too large, we will generate every num_micro_batches/8 ones
    # which means there would be at most 8 configs
    generate_parts = num_micro_batches if num_micro_batches < 8 else 8
    possible_granularities = []

    for i in range(0, num_micro_batches, num_micro_batches//generate_parts):
        # although the first position indicate the stage numbers, there is no difference
        # between specifying for each stage

        # besides, to avoid search space exploration, we assume the behaviour of micro-batches
        # would not interleave
        possible_granularities.append([1, i, 1, num_micro_batches-i, 0])

# TODO: not consider the fine-grained recomp
def GenHeteroConfigs(
        device_type_list,
        device_num_list,
        global_batch_size,
        num_layers,
        num_micro_batches,
        hetero_configs: list
):
    
    devices_info = DevicesInfo(device_type_list=device_type_list, device_num_list=device_num_list)
    hetero_meshes = generate_hetero_meshes(
        devices_info=devices_info, 
        global_batch_size=global_batch_size, 
        num_layers=num_layers)
    for mesh in hetero_meshes:
        pp_stages = sum(mesh[4::5])
        pp_layer_splits = split_layers(num_layers=num_layers, pp_stages=pp_stages)
        for split in pp_layer_splits:
            hetero_config = HeteroConfig(mesh=mesh, 
                                         pp_layer_split=split,
                                         device_types=device_type_list)
            
            hetero_configs.append(hetero_config)
    return

import flagscale.train.theoretical_memory_usage as mem_usg
import analylize_pipeline_time

def report_oom_error(
        memory_capacity_of_devices: list,
        meshes_config: list,
        peak_memory_usage_per_stage: list
):
    stage_index = 0
    for mesh_index, num_stages_in_current_mesh in enumerate(meshes_config[4::5]):
        for i in range(num_stages_in_current_mesh):
            if peak_memory_usage_per_stage[stage_index+i] >= memory_capacity_of_devices[mesh_index]:
                return True
    return False

def gen_memory_model_args_from_mesh_config(
        mesh_config: list
):
    # TODO: build from mesh config, adopt the mem_usg
    args = {}
    return args


def prune_by_memory_model(
        hetero_configs: list,
        num_micro_batches: int,
        memory_capacity_of_devices: list
):
    for mesh_config in hetero_configs:
        peak_memory_usage_per_stage = []
        model_parallel_training_args = gen_memory_model_args_from_mesh_config(mesh_config)
        for stage in range(len(mesh_config)//5):
            peak_activation_memory_usage = mem_usg.compute_activation_memory(args=model_parallel_training_args, num_microbatches=num_micro_batches)
            peak_weight_optimizer_usage = mem_usg.compute_weight_and_optimizer_memory(args=model_parallel_training_args)
            peak_memory_usage = peak_activation_memory_usage + peak_weight_optimizer_usage

            peak_memory_usage_per_stage.append(peak_memory_usage)
        if report_oom_error(memory_capacity_of_devices,
                            mesh_config, 
                            peak_memory_usage_per_stage):
            hetero_configs.remove(mesh_config)

# for test and usage
if __name__ == "__main__":
    device_type_list = ["A", "B"]
    device_num_list = [4, 4]
    global_batch_size = 32
    num_micro_batches = 8
    num_layers = 4
    hetero_configs = []
    memory_capacity_of_devices = [80, 32] # GB

    # generate all possible and legal mesh configs, each element of hetero_configs is a mesh list
    GenHeteroConfigs(
        device_type_list=device_type_list,
        device_num_list=device_num_list,
        global_batch_size=global_batch_size,
        num_layers=num_layers,
        num_micro_batches=num_micro_batches,
        hetero_configs=hetero_configs
    )

    # prunning by memory model
    prune_by_memory_model(
        hetero_configs=hetero_configs,
        num_micro_batches=num_micro_batches,
        memory_capacity_of_devices=memory_capacity_of_devices
    )


    # simulation
    for hetero_config in hetero_configs:
        pp_cost = hetero_config.simulated_time = analylize_pipeline_time.analyze_pp_time(
            scheme="1F1B",
            num_micro_batches=num_micro_batches,
            process_mesh=hetero_config.mesh,
            pp_layers_split=hetero_config.pp_layer_split
        )
        print(f"pipeline cost: {pp_cost}")