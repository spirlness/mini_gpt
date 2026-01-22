import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a mini GPT on character level text."
    )

    # 1. 路径相关
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/tiny.txt",
        help="Path to the training data",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Where to save checkpoints"
    )

    # 2. 训练超参（先放几个最基本的）
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size per step"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Total training steps"
    )

    # 3. 硬件相关
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )

    return parser.parse_args()


def train(args):
    # 打印收到的参数，验证配置系统是否工作
    print("--- Starting Training ---")
    print(f"Data path: {args.data_path}")
    print(f"Device: {args.device}")
    print(f"Hyperparams: Batch={args.batch_size}, LR={args.lr}")

    # 后面我们会在这里写具体的训练逻辑
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
