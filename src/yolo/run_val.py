import os

from ultralytics import YOLO


def launch() -> None:

    seed = 0

    os.environ['WANDB_MODE'] = 'disabled'
    model = YOLO('storage/yolo/models/yolo11n-cls.pt') # uncomment for training
    # model = YOLO('runs/classify/large_all/weights/best.pt') # uncomment for validation

    d = model.train(
        seed=seed,
        data="data/animals/complete", 
        name="test",
        project="runs_yolo",
        epochs=10,
    )
    
    d = model.val(data=f"/data/animals/complete", name='test')
    print(d)


if __name__ == '__main__':
    launch()
