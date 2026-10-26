"""Install the metric observer after W&B initialization rebinds module-level log."""


def install_metric_observer(train_module, observer):
    initialize = train_module.init_wandb

    def initialize_then_observe(*args, **kwargs):
        result = initialize(*args, **kwargs)
        native_log = train_module.wandb.log

        def log(data, step=None, **log_kwargs):
            value = native_log(data, step=step, **log_kwargs)
            observer(data, step)
            return value

        train_module.wandb.log = log
        return result

    train_module.init_wandb = initialize_then_observe
