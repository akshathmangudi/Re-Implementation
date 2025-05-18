import argparse
from entry import run_train, run_test, run_check

def main():
    parser = argparse.ArgumentParser(prog="refrakt", description="Refrakt CLI Interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # TRAIN
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("model", type=str)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--device", type=str, default="cuda")

    # TEST
    test_parser = subparsers.add_parser("test", help="Test a model")
    test_parser.add_argument("model", type=str)
    test_parser.add_argument("--batch-size", type=int, default=128)
    test_parser.add_argument("--device", type=str, default="cuda")

    # CHECK
    check_parser = subparsers.add_parser("check", help="Run system and test checks")

    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "test":
        run_test(args)
    elif args.command == "check":
        run_check()
