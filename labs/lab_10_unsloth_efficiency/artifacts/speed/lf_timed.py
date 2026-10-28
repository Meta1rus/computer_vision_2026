import json, sys, threading, time
import torch, yaml
from transformers import TrainerCallback
from llamafactory.train.tuner import run_exp

cfg = yaml.safe_load(open(sys.argv[1]))
marks, peak, done = [], {"allocated": 0, "driver": 0}, threading.Event()

def sample():
    while not done.is_set():
        peak["allocated"] = max(peak["allocated"], torch.mps.current_allocated_memory())
        peak["driver"] = max(peak["driver"], torch.mps.driver_allocated_memory())
        time.sleep(0.05)

class StepTimer(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        torch.mps.synchronize(); marks.append([0, time.perf_counter()])
    def on_step_end(self, args, state, control, **kwargs):
        torch.mps.synchronize(); marks.append([state.global_step, time.perf_counter()])

threading.Thread(target=sample, daemon=True).start()
run_exp(cfg, callbacks=[StepTimer()])
done.set()
json.dump({"marks": marks, "peak_allocated_gb": peak["allocated"] / 1e9, "peak_driver_gb": peak["driver"] / 1e9}, open(sys.argv[2], "w"))
