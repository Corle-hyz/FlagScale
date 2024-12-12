# Environment
Begin at the root path of `FlagScale` repository:
```
conda activate flagscale
cd flagscale/auto_tuner/simulator/torch_collective_extension/custom_backend/
python setup.py develop
cd ../../
```

# Setup
Set necessary parameters in `config_gen.py`. For example:
```
device_type_list = ["A", "B"]
device_num_list = [4, 4]
global_batch_size = 32
num_micro_batches = 8
num_layers = 4
```
# Run a Task
Start the auto-tuning: 
```
python config_gen.py
```