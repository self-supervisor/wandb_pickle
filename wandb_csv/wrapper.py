import datetime
import pickle
import socket
from typing import List

import wandb


class WandbCSV:
    def __init__(
        self,
        config: dict,
        log_dir: str = "wandb_csv",
        log_prefixes: List[str] = ["training", "evaluating"],
        backup: bool = False,
    ):
        assert type(config) == dict
        self.log_prefixes = log_prefixes
        self.config = config
        self.log_dir = log_dir
        self.metrics = {}
        for k in self.log_prefixes:
            self.metrics[k] = []
        self.backup = backup
        self.run_ID = socket.gethostname() + str(datetime.datetime.now())

    def make_run_ID(self) -> str:
        now = datetime.now()
        timestamp = now.strftime("_%Y-%m-%d_%H:%M:%S")
        hostname = socket.gethostname()
        ID = hostname + timestamp
        ID.replace(" ", "_")
        return ID

    def log(self, log_prefix: str, metrics_to_log: dict) -> None:
        assert (
            log_prefix in self.log_prefixes
        ), f"log prefix must be in {self.log_prefixes} but is {log_prefix}"
        self.metrics[log_prefix].append(metrics_to_log)
        metrics_with_prefix = {}
        for key in metrics_to_log.keys():
            new_key = log_prefix + "/" + key
            metrics_with_prefix[new_key] = metrics_to_log[key]
        wandb.log(metrics_with_prefix)

    def save(self):
        save_str = f"{self.log_dir}/wandb_logger_{self.run_ID}.pkl"
        save_str.replace(" ", "")
        with open(save_str, "wb") as f:
            pickle.dump(self, f)

        if self.backup:
            self.backup_files()

    def backup_files(self):
        raise NotImplementedError("Backups have not been implemented yet")
