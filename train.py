import argparse
import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, required=True)
    # args = parser.parse_args()

    model = YOLO('./ultralytics/cfg/models/11-RGBT/yolo11-RGBT-midfusion.yaml')

    model.train(data='./M3FD-rgbt.yaml',
                cache=False,
                imgsz=640,
                epochs=3,
                batch=4,
                close_mosaic=20,
                workers=2,
                optimizer='SGD',
                lr0=0.01,
                lrf=0.01,
                device='0',
                use_simotm="RGBT",
                channels=4,
                project='runs/1',
                name='train',
                )
