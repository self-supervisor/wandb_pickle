# %%
import wandb
import numpy as np

from wandb_csv.wrapper import (
    WandbPickle,
    get_files,
    line_plot_a_metric,
    line_plot_mean_and_stderr,
    load_file,
)

wandb.init(mode="dryrun")

# %%
wandb_csv1 = WandbPickle(
    config={"learning_rate": 0.01, "seed": 1}, log_dir="example_logs"
)


for i in range(100):
    wandb_csv1.log("training", {"loss": 0.01 * i, "accuracy": 1 - 0.01 * i})
wandb_csv1.save()
wandb.finish()
# %%
wandb_csv1 = load_file(get_files("example_logs/*.pkl")[0])
line_plot_a_metric(wandb_csv1, "training", "loss", "accuracy")
# %%
wandb.init(mode="dryrun")
wandb_csv2 = WandbPickle(
    config={"learning_rate": 0.01, "seed": 2}, log_dir="example_logs"
)
for i in range(100):
    wandb_csv2.log("training", {"loss": 0.01 * i, "accuracy": 0.9 - 0.01 * i})
wandb_csv2.save()
wandb.finish()

wandb.init(mode="dryrun")


wandb_csv3 = WandbPickle(
    config={"learning_rate": 0.01, "seed": 3}, log_dir="example_logs"
)
for i in range(100):
    wandb_csv3.log("training", {"loss": 0.1 * i, "accuracy": 0.8 - 0.01 * i})
wandb_csv3.save()
wandb.finish()
# %%
all_file_paths = get_files("example_logs/*.pkl")
loaded_files = [load_file(a_file) for a_file in all_file_paths]
line_plot_mean_and_stderr(loaded_files, "training", "loss", "accuracy")
# %%
