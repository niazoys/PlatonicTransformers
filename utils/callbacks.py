import time
from typing import Any, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch

try:  # Optional dependency used for system memory telemetry.
    import psutil  # type: ignore[import-not-found]
except ModuleNotFoundError:
    psutil = None  # type: ignore[assignment]


class TimerCallback(pl.Callback):
    """Log end-to-end training and test inference durations."""

    def __init__(self) -> None:
        super().__init__()
        self.total_training_start_time = 0.0
        self.epoch_start_time = 0.0
        self.test_inference_time = 0.0

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.total_training_start_time = time.time()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        total_training_time = (time.time() - self.total_training_start_time) / 60
        trainer.logger.experiment.log({"Total Training Time (min)": total_training_time})

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.epoch_start_time = time.time()

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.test_inference_time = (time.time() - self.epoch_start_time) / 60
        trainer.logger.experiment.log({"Test Inference Time (min)": self.test_inference_time})


class StopOnPersistentDivergence(pl.Callback):
    """Stop training when a monitored metric exceeds a threshold for too long."""

    def __init__(
        self,
        monitor: str = "val MAE",
        threshold: float = 0.8,
        patience: int = 3,
        grace_epochs: int = 10,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(monitor, str) or not monitor:
            raise ValueError("Argument `monitor` must be a non-empty string.")
        if not isinstance(threshold, (int, float)):
            raise ValueError("Argument `threshold` must be a number.")
        if not isinstance(patience, int) or patience < 1:
            raise ValueError("Argument `patience` must be an integer >= 1.")
        if not isinstance(grace_epochs, int) or grace_epochs < 0:
            raise ValueError("Argument `grace_epochs` must be a non-negative integer.")

        self.monitor = monitor
        self.threshold = threshold
        self.patience = patience
        self.grace_epochs = grace_epochs
        self.verbose = verbose

        self.consecutive_exceeds_count = 0
        self.stopped_epoch = 0

    def _check_metric(self, trainer: pl.Trainer) -> Optional[float]:
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            metrics = trainer.logged_metrics

        if self.monitor in metrics:
            metric_val = metrics[self.monitor]
            if hasattr(metric_val, "item"):
                return metric_val.item()
            try:
                return float(metric_val)
            except (ValueError, TypeError):
                if self.verbose:
                    print(
                        f"{self.__class__.__name__}: Warning: Metric '{self.monitor}' has non-convertible type {type(metric_val)}."
                    )
                return None
        return None

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        current_epoch = trainer.current_epoch

        if current_epoch < self.grace_epochs:
            if self.verbose and current_epoch == 0:
                print(
                    f"{self.__class__.__name__}: In grace period (first {self.grace_epochs} epochs). "
                    f"Divergence check for '{self.monitor}' (threshold > {self.threshold}) is inactive."
                )
            return

        current_metric_value = self._check_metric(trainer)

        if current_metric_value is None:
            if self.verbose and current_epoch == self.grace_epochs:
                print(
                    f"{self.__class__.__name__}: Metric '{self.monitor}' not found in logs at epoch {current_epoch} "
                    f"(when divergence check became active). "
                    f"Available callback_metrics: {list(trainer.callback_metrics.keys())}. "
                    f"Available logged_metrics: {list(trainer.logged_metrics.keys())}."
                )
            return

        status_message = (
            f"{self.__class__.__name__} at epoch {current_epoch}: '{self.monitor}' = {current_metric_value:.4f}. "
            f"Threshold: > {self.threshold}. Consecutive exceeds: {self.consecutive_exceeds_count}. Patience: {self.patience}."
        )

        if current_metric_value > self.threshold:
            self.consecutive_exceeds_count += 1
            if self.verbose:
                print(f"{status_message} Metric EXCEEDED threshold.")
        else:
            if self.verbose and self.consecutive_exceeds_count > 0:
                print(f"{status_message} Metric NOT ABOVE threshold. Resetting consecutive count.")
            self.consecutive_exceeds_count = 0

        if self.consecutive_exceeds_count >= self.patience:
            self.stopped_epoch = current_epoch
            trainer.should_stop = True
            if self.verbose:
                print(
                    f"\n{self.__class__.__name__}: Stopping training at epoch {self.stopped_epoch} "
                    f"because '{self.monitor}' ({current_metric_value:.4f}) "
                    f"was above divergence threshold ({self.threshold}) "
                    f"for {self.consecutive_exceeds_count} consecutive epochs (patience={self.patience}) "
                    f"after grace period of {self.grace_epochs} epochs."
                )

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.stopped_epoch > 0 and self.verbose:
            print(
                f"{self.__class__.__name__}: Training ended. Stopped early at epoch {self.stopped_epoch} "
                f"due to persistent divergence of '{self.monitor}'."
            )


class TrainingTimerCallback(pl.Callback):
    """Measure forward and full training cycle timings."""

    def __init__(
        self,
        num_epochs_to_measure: int = 3,
        forward_function: str = "forward",
        units: str = "ms",
    ) -> None:
        super().__init__()
        self.num_epochs_to_measure = num_epochs_to_measure
        self.forward_function = forward_function
        self.units = units
        self.multiplier = 1000 if units == "ms" else 1

        self.validation_forward_times: List[float] = []
        self.training_full_times: List[float] = []

        self.measuring = False
        self.epochs_measured = 0

        self.current_val_forward_times: List[float] = []
        self.current_train_full_times: List[float] = []

        self.original_forward_function: Optional[Any] = None
        self.batch_start_time: Optional[float] = None
        self.function_wrapped = False

        self.in_validation = False
        self.in_training = False

    def _should_measure(self) -> bool:
        return self.epochs_measured < self.num_epochs_to_measure

    def _wrap_forward_function(self, pl_module: pl.LightningModule) -> None:
        if not hasattr(pl_module, self.forward_function):
            print(f"Warning: {self.forward_function} method not found in module")
            return

        if self.function_wrapped:
            return

        self.original_forward_function = getattr(pl_module, self.forward_function)
        self.function_wrapped = True

        def timed_forward(*args: Any, **kwargs: Any) -> Any:
            if self.measuring and self.in_validation:
                start_time = time.perf_counter()
                result = self.original_forward_function(*args, **kwargs)
                end_time = time.perf_counter()
                forward_time = (end_time - start_time) * self.multiplier
                self.current_val_forward_times.append(forward_time)
                return result
            return self.original_forward_function(*args, **kwargs)

        setattr(pl_module, self.forward_function, timed_forward)

    def _restore_forward_function(self, pl_module: pl.LightningModule) -> None:
        if self.original_forward_function is not None:
            setattr(pl_module, self.forward_function, self.original_forward_function)
            self.function_wrapped = False

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._should_measure():
            self.current_train_full_times = []
            self.measuring = True
            self.in_training = True
            self._wrap_forward_function(pl_module)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.measuring and self.in_training:
            self.batch_start_time = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.measuring and self.in_training and self.batch_start_time is not None:
            batch_end_time = time.perf_counter()
            full_training_time = (batch_end_time - self.batch_start_time) * self.multiplier
            self.current_train_full_times.append(full_training_time)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.measuring and self.in_training:
            self.in_training = False
            if self.current_train_full_times:
                self.training_full_times.extend(self.current_train_full_times)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._should_measure():
            self.current_val_forward_times = []
            self.in_validation = True

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._should_measure() and self.in_validation:
            self.in_validation = False
            if self.current_val_forward_times:
                self.validation_forward_times.extend(self.current_val_forward_times)
            self.epochs_measured += 1
            if self.epochs_measured >= self.num_epochs_to_measure:
                self._log_final_results(pl_module)
                self.measuring = False
                self._restore_forward_function(pl_module)

    def _log_final_results(self, pl_module: pl.LightningModule) -> None:
        if self.training_full_times:
            avg_training_time = np.mean(self.training_full_times)
            std_training_time = np.std(self.training_full_times)
            pl_module.log(
                f"avg_forward_backward_optimizer_time_[{self.units}]",
                avg_training_time,
                prog_bar=True,
                logger=True,
            )
            print(
                f"Average (forward+backward+optimizer) time over {self.epochs_measured} epochs: "
                f"{avg_training_time:.2f} ± {std_training_time:.2f} {self.units}"
            )

        if self.validation_forward_times:
            avg_validation_time = np.mean(self.validation_forward_times)
            std_validation_time = np.std(self.validation_forward_times)
            pl_module.log(
                f"avg_{self.forward_function}_time_[{self.units}]",
                avg_validation_time,
                prog_bar=True,
                logger=True,
            )
            print(
                f"Average {self.forward_function} time over {self.epochs_measured} epochs: "
                f"{avg_validation_time:.2f} ± {std_validation_time:.2f} {self.units}"
            )


class MemoryMonitorCallback(pl.Callback):
    """Log system and GPU memory usage at a fixed batch frequency."""

    def __init__(self, log_frequency: int = 500) -> None:
        super().__init__()
        self.log_frequency = log_frequency

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.log_frequency == 0:
            if psutil is None:
                raise ModuleNotFoundError(
                    "MemoryMonitorCallback requires the optional 'psutil' dependency."
                )
            process = psutil.Process()
            ram_gb = process.memory_info().rss / (1024 * 1024 * 1024)
            pl_module.log("system_memory_gb", ram_gb, prog_bar=False)

            if torch.cuda.is_available():
                gpu_gb = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
                pl_module.log("gpu_memory_gb", gpu_gb, prog_bar=False)


class NaNDetectorCallback(pl.callbacks.Callback):
    """Monitor for NaN values in losses, gradients, parameters, and module outputs."""

    def __init__(
        self,
        check_gradients: bool = True,
        check_parameters: bool = True,
        check_loss: bool = True,
        terminate_on_nan: bool = True,
        log_module_outputs: bool = True,
    ) -> None:
        super().__init__()
        self.check_gradients = check_gradients
        self.check_parameters = check_parameters
        self.check_loss = check_loss
        self.terminate_on_nan = terminate_on_nan
        self.log_module_outputs = log_module_outputs
        self.nan_detected = False
        self.hooks: List[Any] = []

    def _register_hooks(self, pl_module: pl.LightningModule) -> None:
        if not self.log_module_outputs:
            return

        def hook_fn(module: torch.nn.Module, inputs: Any, output: Any, name: str) -> None:
            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any()
            elif isinstance(output, (tuple, list)):
                has_nan = any(torch.is_tensor(x) and torch.isnan(x).any() for x in output)
            else:
                has_nan = False

            if has_nan:
                self.nan_detected = True
                message = f"NaN detected in output of module: {name}"
                print(f"\n{'=' * 80}\n{message}\n{'=' * 80}\n")

                if self.terminate_on_nan:
                    raise ValueError(message)

        for name, module in pl_module.named_modules():
            if name:
                hook = module.register_forward_hook(
                    lambda mod, inp, outp, name=name: hook_fn(mod, inp, outp, name)
                )
                self.hooks.append(hook)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._register_hooks(pl_module)

    def _check_tensor_for_nan(self, tensor: Any, tensor_name: str) -> bool:
        if tensor is None or not torch.is_tensor(tensor):
            return False

        if torch.isnan(tensor).any():
            self.nan_detected = True
            message = f"NaN detected in {tensor_name}"
            print(f"\n{'=' * 80}\n{message}\n{'=' * 80}\n")

            if self.terminate_on_nan:
                raise ValueError(message)
            return True
        return False

    def on_before_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule, loss: torch.Tensor) -> None:
        if self.check_loss:
            self._check_tensor_for_nan(loss, "loss")

    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.check_gradients:
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    self._check_tensor_for_nan(param.grad, f"gradient of {name}")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.check_parameters:
            for name, param in pl_module.named_parameters():
                self._check_tensor_for_nan(param.data, f"parameter {name}")

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
