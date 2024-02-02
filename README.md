# Wandb pickle 🥒
# Easy Custom Plots with Wandb 😊

Wandb is great for visualisations and debugging of performance in machine learning experiments. However, if you want to extract that data off of wandb it can be very awkward. In particular, if you want to make custom matplotlib visualisations of the graphs you see on wandb it requires significant effort to scrape the data off of their servers. This is a wrapper that makes that process easier, by logging your wandb metrics to a local pickle from which data can be accessed very quickly for plotting.

## 🚀 Quick Start

```python
import wandb
from wandb_pickle.wrapper import WandbPickle, line_plot_a_metric, line_plot_mean_and_stderr, load_file, get_files

# Initialize wandb
wandb.init(mode="dryrun")

# Logging with WandbPickle
wandb_pickle1 = WandbPickle(config={"learning_rate": 0.01, "seed": 1}, log_dir="example_logs")
for i in range(100):
    wandb_pickle1.log("training", {"loss": 0.01 * i, "accuracy": 1 - 0.01 * i})
wandb_pickle1.save()
wandb.finish()
```
This will save a pickle file locally of your logged metrics. Then you can load the metrics you want to plot and plot them very easily.

```python
# Load data and plot a single metric
wandb_pickle1 = load_file(get_files("example_logs/*.pkl")[0])
line_plot_a_metric(wandb_pickle1, "training", "loss", "accuracy")
```

Where line_plot_a_metric is a very simple function here.

If you want to plot the mean and stderr of a bunch of runs (as you often want to in reinforcement learning), you can do the following.

```python
# Logging different additional runs
wandb.init(mode="dryrun")
wandb_pickle2 = WandbPickle(config={"learning_rate": 0.01, "seed": 2}, log_dir="example_logs")
for i in range(100):
    wandb_pickle2.log("training", {"loss": 0.01 * i, "accuracy": 0.9 - 0.01 * i})
wandb_pickle2.save()
wandb.finish()

wandb.init(mode="dryrun")
wandb_pickle3 = WandbPickle(config={"learning_rate": 0.01, "seed": 3}, log_dir="example_logs")
for i in range(100):
    wandb_pickle3.log("training", {"loss": 0.1 * i, "accuracy": 0.8 - 0.01 * i})
wandb_pickle3.save()
wandb.finish()

# Load all data and plot mean and standard error
all_file_paths = get_files("example_logs/*.pkl")
loaded_files = [load_file(a_file) for a_file in all_file_paths]
line_plot_mean_and_stderr(loaded_files, "training", "loss", "accuracy")
```

