from os import path
from time import time
from typing import Protocol
from dataclasses import dataclass, field


@dataclass
class Metrics:
    loss: float = 0
    epe: float = 0
    e3p: float = 0
    items: int = 0
    iters: int = 0
    start_time: float = field(default_factory=time)
    end_time: float = 0

    def __str__(self):
        res = f"Time taken: {self.time_taken:.2f}s\n"
        res += f"Average loss: {self.avg_loss:.4f}\n"
        res += f"Average endpoint error: {self.avg_epe:.4f}\n"
        res += f"Average 3 pixel error: {self.avg_e3p:.4f}"
        return res

    def add(self, loss, epe, e3p, items, iters):
        self.loss += loss
        self.epe += epe
        self.e3p += e3p
        self.items += items
        self.iters += iters

    def end(self):
        self.end_time = time()

    @property
    def time_taken(self):
        return self.end_time - self.start_time if self.end_time else -1

    @property
    def avg_epe(self):
        return self.epe / self.items if self.items else -1

    @property
    def avg_e3p(self):
        return self.e3p / self.items if self.items else -1

    @property
    def avg_loss(self):
        return self.loss / self.items if self.items else -1


class Logger(Protocol):
    def append(self, logtype, epoch, training_metrics, test_metric=None):
        pass


class TrainingLogger:
    def __init__(
        self, filename: str, save_filename: str, dataset_name: str, learning_rate: str
    ):
        self.filename = filename
        self.prefix = f"{save_filename},{dataset_name},{learning_rate},"

    def append(
        self, logtype, epoch, training_metrics: Metrics, test_metric: Metrics = None
    ):
        if test_metric is None:
            test_metric = Metrics()

        log = (
            self.prefix
            + f"{epoch},{logtype},{training_metrics.iters},"
            + f"{self.metric_to_log(training_metrics)},"
            + f"{self.metric_to_log(test_metric)}"
            + "\n"
        )
        with open(self.filename, "a+") as f:
            f.write(log)

    def metric_to_log(self, metric: Metrics) -> str:
        m = metric
        return f"{m.time_taken},{m.avg_loss},{m.avg_epe},{m.avg_e3p}"


class DummyTrainingLogger:
    def __init__(self):
        pass

    def append(self, logtype, epoch, training_metrics, test_metric=None):
        pass


def create_logger(log_file, save_file, dataset_name, learning_rate) -> Logger:
    if not log_file:
        return DummyTrainingLogger()
    if path.isdir(log_file):
        raise ValueError("Given log file path is a directory and save is not defined")

    save_filename = "NoSave"
    if save_file:
        save_filename = path.basename(save_file)

    return TrainingLogger(log_file, save_filename, dataset_name, learning_rate)
