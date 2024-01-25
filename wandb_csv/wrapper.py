import glob
import pickle
import socket
from datetime import datetime
from typing import List, Tuple

import numpy as np
import wandb


class WandbCSV:
    """A class for logging and saving wandb runs, but with easy customisable local plotting."""

    def __init__(
        self,
        config: dict,
        log_dir: str = "wandb_csv",
        log_prefixes: List[str] = ["training", "evaluating"],
        backup: bool = False,
    ):
        """
        Initialize the WandbCSV object.

        Args:
            config (dict): Configuration dictionary.
            log_dir (str, optional): Directory for logging. Defaults to "wandb_csv".
            log_prefixes (List[str], optional): List of prefixes for logging. Defaults to ["training", "evaluating"].
            backup (bool, optional): Whether to backup files. Defaults to False.
        """
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
        """
        Generate a unique run ID based on the current timestamp and hostname.

        The run ID is a string in the format "hostname_YYYY-MM-DD_HH:MM:SS", where
        "hostname" is the name of the host machine, "YYYY-MM-DD" is the current date,
        and "HH:MM:SS" is the current time.

        Returns:
            str: The generated run ID.
        """

    def log(self, log_prefix: str, metrics_to_log: dict) -> None:
        """
        Log metrics with a specified prefix.

        This function logs the provided metrics under the specified prefix. The metrics are stored in the `self.metrics` dictionary
        and also logged to Weights & Biases using the `wandb.log` function.

        Args:
            log_prefix (str): The prefix for the metrics. Must be in `self.log_prefixes`.
            metrics_to_log (dict): The metrics to log. The keys are the metric names and the values are the metric values.

        Raises:
            AssertionError: If `log_prefix` is not in `self.log_prefixes`.

        Returns:
            None
        """
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
        """
        Save the current WandbCSV object to a pickle file.

        The object is saved to a file named "wandb_logger_{run_ID}.pkl" in the directory specified by `self.log_dir`.
        Spaces in the filename are removed. If `self.backup` is True, the `backup_files` method is also called.

        Returns:
            None
        """
        save_str = f"{self.log_dir}/wandb_logger_{self.run_ID}.pkl"
        save_str.replace(" ", "")
        with open(save_str, "wb") as f:
            pickle.dump(self, f)

        if self.backup:
            self.backup_files()

    def backup_files(self) -> None:
        raise NotImplementedError("Backups have not been implemented yet")


def get_files(path: str) -> List[str]:
    """
    Globs all files in a directory.

    Args:
        path (str): Path to the directory.

    Returns:
        List[str]: List of file paths.
    """
    return glob.glob(path)


def load_file(file_path: str):
    """
    Load a file using pickle.

    Args:
        file_path (str): Path to the file.

    Returns:
        The content of the file.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def index_data(
    list_of_files: List[str], x_quantity: float
) -> dict[Tuple[float, int], List[float]]:
    """
    Index data from a list of files.

    Args:
        list_of_files (List[str]): List of file paths.
        x_quantity (float): Quantity for indexing.

    Returns:
        dict[Tuple[float, int], List[float]]: Indexed data.
    """
    data_index = {}
    for a_file in list_of_files:
        data = load_file(a_file)
        key = (data.config[x_quantity], data.config["seed"])
        value = data
        data_index[key] = value
    return data_index


def calculate_statistics(reward_list: List[float]) -> Tuple[float, float]:
    """
    Calculate the mean and standard error of a list of rewards.

    Args:
        reward_list (List[float]): List of rewards.

    Returns:
        Tuple[float, float]: Mean and standard error of the rewards.
    """
    mean_reward = np.mean(reward_list)
    std_error = np.std(reward_list) / np.sqrt(len(reward_list))
    return mean_reward, std_error


def extract_metric(
    wandb_csv_prefix: str, metric_name: str, run: WandbCSV
) -> np.ndarray:
    """
    Extract a metric from a WandbCSV object.

    Args:
        wandb_csv_prefix (str): Prefix for the metric.
        metric_name (str): Name of the metric.
        run (WandbCSV): WandbCSV object.

    Returns:
        np.ndarray: Array of metric values.
    """
    return np.array(
        [
            run.metrics[wandb_csv_prefix][metric_name][i]
            for i in range(len(run.metrics[wandb_csv_prefix][metric_name]))
        ]
    )


def calculate_statistics_on_list_of_lists(list_of_lists: List) -> Tuple[List, List]:
    """
    Calculate statistics on a list of lists. Currently only supports mean and standard error.

    Args:
        list_of_lists (List): List of lists of values.

    Returns:
        Tuple[List, List]: Mean and standard error for each list of values.
    """
    transposed_data = list(zip(*list_of_lists))

    mean_list = []
    stderr_list = []
    for metric_values in transposed_data:
        mean, stderr = calculate_statistics(metric_values)
        mean_list.append(mean)
        stderr_list.append(stderr)

    return mean_list, stderr_list


def filter_csvs(csv: List[WandbCSV], filter_key, filter_value) -> List[WandbCSV]:
    """
    Filter a list of WandbCSV objects based on a key-value pair.

    Args:
        csv (List[WandbCSV]): List of WandbCSV objects.
        filter_key: Key for filtering.
        filter_value: Value for filtering.

    Returns:
        List[WandbCSV]: Filtered list of WandbCSV objects.
    """
    return [c for c in csv if c.config[filter_key] == filter_value]
