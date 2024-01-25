import numpy as np
import pytest
import wandb

from wandb_csv.wrapper import (
    WandbCSV,
    calculate_statistics,
    calculate_statistics_on_list_of_lists,
)

wandb.init(mode="dryrun")


@pytest.fixture
def wandb_csv_instance():
    config = {"param1": "value1", "param2": "value2"}
    log_dir = "test_dir"
    return WandbCSV(config, log_dir)


def test_make_run_ID(wandb_csv_instance):
    run_ID = wandb_csv_instance.make_run_ID()
    assert isinstance(run_ID, str)


def test_log(wandb_csv_instance):
    wandb_csv_instance.log("training", {"loss": 0.5})
    assert "loss" in wandb_csv_instance.metrics["training"]
    assert wandb_csv_instance.metrics["training"]["loss"] == [0.5]


def test_calculate_statistics():
    reward_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean_reward, std_error = calculate_statistics(reward_list)
    assert mean_reward == np.mean(reward_list)
    assert std_error == np.std(reward_list) / np.sqrt(len(reward_list))


def test_calculate_statistics_on_list_of_lists():
    list_of_lists = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    mean_list, stderr_list = calculate_statistics_on_list_of_lists(list_of_lists)
    assert mean_list == [np.mean([1.0, 4.0]), np.mean([2.0, 5.0]), np.mean([3.0, 6.0])]
    assert stderr_list == [
        np.std([1.0, 4.0]) / np.sqrt(2),
        np.std([2.0, 5.0]) / np.sqrt(2),
        np.std([3.0, 6.0]) / np.sqrt(2),
    ]
