import logging
import os
import sys
from typing import List, Optional


class RefraktLogger:
    def __init__(
        self,
        log_dir: str,
        log_file: str = "refrakt.log",
        log_types: Optional[List[str]] = None,
        console: bool = False,
        debug: bool = False,
    ):
        self.log_dir = os.path.abspath(log_dir)  # Ensure absolute path
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, log_file)
        self.log_types = log_types or []
        self.console = console
        self.wandb_run = None
        self.tb_writer = None
        self.debug_enabled = debug

        self.logger = logging.getLogger("refrakt")
        level = logging.DEBUG if debug else logging.INFO
        self.logger.setLevel(level)

        # KEY FIX: Only configure handlers if none exist
        if not self.logger.hasHandlers():
            self._setup_handlers(level)

        # CRITICAL FIX: Prevent propagation to root logger
        self.logger.propagate = False

        # Setup experiment trackers
        if "wandb" in self.log_types:
            self._setup_wandb()
        if "tensorboard" in self.log_types:
            self._setup_tensorboard()

    def _setup_handlers(self, level):
        """Setup handlers only once"""
        # Clear any existing handlers (just in case)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(file_handler)

        if self.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(console_handler)

    def _setup_wandb(self):
        try:
            import wandb

            self.wandb_run = wandb.init(project="refrakt", dir=self.log_dir)
            self.info("Weights & Biases initialized")
        except ImportError:
            self.error("wandb not installed. Skipping WandB setup")
        except Exception as e:
            self.error(f"WandB initialization failed: {str(e)}")

    def _setup_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = os.path.join(self.log_dir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            print(f"[DEBUG] Creating SummaryWriter at {tb_dir}")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            self.info(f"TensorBoard initialized at {tb_dir}")
        except Exception as e:
            self.error(f"TensorBoard initialization failed: {str(e)}")

    def log_metrics(self, metrics: dict, step: int):
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)

    def log_config(self, config: dict):
        if self.wandb_run:
            self.wandb_run.config.update(config)

        if self.tb_writer:
            from torch.utils.tensorboard.summary import hparams

            try:
                filtered_config = flatten_and_filter_config(config)
                exp, ssi, sei = hparams(filtered_config, {})
                self.tb_writer.file_writer.add_summary(exp)
                self.tb_writer.file_writer.add_summary(ssi)
                self.tb_writer.file_writer.add_summary(sei)
                self.info("Logged filtered config to TensorBoard hparams")
            except Exception as e:
                self.error(f"Failed to log hparams to TensorBoard: {str(e)}")

    def info(self, msg: str):
        self.logger.info(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def debug(self, msg):
        if self.debug_enabled:
            self.logger.debug(msg)

    def close(self):
        # Close handlers to prevent resource leaks
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            self.wandb_run.finish()
