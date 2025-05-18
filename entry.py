from trainer import Trainer
from registry import get_model, get_dataset, get_loss, get_optimizer

def run_train(args):
    model = get_model(args.model)
    train_loader, val_loader = get_dataset(args.model, args.batch_size)
    loss_fn = get_loss(args.model)
    optimizer = get_optimizer(model, lr=args.lr)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=args.device,
    )
    trainer.train(num_epochs=args.epochs)


def run_test(args):
    model = get_model(args.model, pretrained=True)
    _, val_loader = get_dataset(args.model, args.batch_size)
    loss_fn = get_loss(args.model)

    trainer = Trainer(
        model=model,
        train_loader=None,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=None,
        device=args.device,
    )
    trainer.evaluate()
