import glob
import pickle
import socket
from datetime import datetime
from typing import List, Tuple

import numpy as np
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
            self.metrics[k] = {}
        self.backup = backup
        self.run_ID = self.make_run_ID()

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
        if len(self.metrics[log_prefix]) == 0:
            for key in metrics_to_log.keys():
                self.metrics[log_prefix][key] = [metrics_to_log[key]]
        else:
            for key in metrics_to_log.keys():
                self.metrics[log_prefix][key].append(metrics_to_log[key])

        metrics_with_prefix = {}
        for key in metrics_to_log.keys():
            new_key = log_prefix + "/" + key
            metrics_with_prefix[new_key] = metrics_to_log[key]
        wandb.log(metrics_with_prefix)

    def save(self) -> None:
        save_str = f"{self.log_dir}/wandb_logger_{self.run_ID}.pkl"
        save_str.replace(" ", "")
        with open(save_str, "wb") as f:
            pickle.dump(self, f)

        if self.backup:
            self.backup_files()

    def backup_files(self) -> None:
        raise NotImplementedError("Backups have not been implemented yet")


def get_files(path: str) -> List[str]:
    return glob.glob(path)


def load_file(file_path: str):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def index_data(
    list_of_files: List[str], x_quantity: float
) -> dict[Tuple[float, int], List[float]]:
    data_index = {}
    for a_file in list_of_files:
        data = load_file(a_file)
        key = (data.config[x_quantity], data.config["seed"])
        value = data
        data_index[key] = value
    return data_index


def calculate_statistics(reward_list: List[float]) -> Tuple[float, float]:
    mean_reward = np.mean(reward_list)
    std_error = np.std(reward_list) / np.sqrt(len(reward_list))
    return mean_reward, std_error


def extract_metric(
    wandb_csv_prefix: str, metric_name: str, run: WandbCSV
) -> np.ndarray:
    return np.array(
        [
            run.metrics[wandb_csv_prefix][metric_name][i]
            for i in range(len(run.metrics[wandb_csv_prefix][metric_name]))
        ]
    )


def calculate_statistics_on_list_of_lists(list_of_lists: List) -> Tuple[List, List]:
    transposed_data = list(zip(*list_of_lists))

    mean_list = []
    stderr_list = []
    for metric_values in transposed_data:
        mean, stderr = calculate_statistics(metric_values)
        mean_list.append(mean)
        stderr_list.append(stderr)

    return mean_list, stderr_list
