import argparse
import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO11 RGBT Training Script')

    # Model and dataset config
    parser.add_argument('--mode', type=str, default='midfusion', help='Training mode')
    parser.add_argument('--dataset', type=str, default='M3FD', help='Dataset name')

    # Training parameters
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--close_mosaic', type=int, default=20, help='Disable mosaic augmentation after N epochs')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--cache', action='store_true', help='Cache dataset in memory')

    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer type')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate factor')

    # Device and output
    parser.add_argument('--device', type=str, default='0', help='CUDA device ID')
    parser.add_argument('--use_simotm', type=str, default='RGBT', help='Modality type')
    parser.add_argument('--channels', type=int, default=4, help='Number of input channels')
    parser.add_argument('--project', type=str, default=None, help='Project save directory (default: runs/{mode})')
    parser.add_argument('--name', type=str, default='train', help='Experiment name')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Select model config based on mode
    model_yaml = f'./ultralytics/cfg/models/11-RGBT/yolo11-RGBT-{args.mode}.yaml'
    model = YOLO(model_yaml)
    print(f'Using model config: {model_yaml}')

    # Select data config based on dataset
    if args.dataset == 'M3FD':
        data_yaml = './M3FD-rgbt.yaml'

    # Set default project path if not specified
    project = args.project if args.project else f'runs/{args.mode}'

    model.train(
        data=data_yaml,
        cache=args.cache,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        close_mosaic=args.close_mosaic,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        device=args.device,
        use_simotm=args.use_simotm,
        channels=args.channels,
        project=project,
        name=args.name,
    )
