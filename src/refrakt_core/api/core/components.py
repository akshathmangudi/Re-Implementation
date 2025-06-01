class ModelComponents:
    """Container for model-related components"""
    def __init__(self, model, loss_fn, optimizer, scheduler=None, device="cuda"):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

