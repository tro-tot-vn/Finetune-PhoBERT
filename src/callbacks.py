import json, os
from transformers import TrainerCallback

class JsonLossLogger(TrainerCallback):
    """
    Ghi loss (train/eval) theo step/epoch vào file JSON để visualize.
    Mỗi dòng là một JSON object (NDJSON).
    """
    def __init__(self, log_path="out/metrics/train_log.ndjson"):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload = {
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            **{k: float(v) for k, v in logs.items() if isinstance(v, (int, float))},
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
