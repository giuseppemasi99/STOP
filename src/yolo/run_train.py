import argparse
import json
import os

import numpy as np
from ultralytics import YOLO


def launch(args) -> None:
    os.environ['WANDB_MODE'] = 'disabled'

    DATA_COMPLETE = 'data/animals/complete'

    seed = args.seed
    size = args.size
    n_classes = args.n_classes

    np.random.seed(seed)

    all_animals = os.listdir(DATA_COMPLETE + '/train')
    if n_classes == len(all_animals):
        data = DATA_COMPLETE
    else:
        DATA_SUBSET = 'data/animals/subset'
        random_indices = np.random.choice(range(len(all_animals)), n_classes, replace=False)
        selected_animals = [all_animals[i].replace(' ', '\\ ') for i in random_indices]
        print(selected_animals)
        os.makedirs(f'{DATA_SUBSET}/train', exist_ok=True)
        os.makedirs(f'{DATA_SUBSET}/val', exist_ok=True)
        for animal in selected_animals:
            os.system(f'cp -r {DATA_COMPLETE}/train/{animal} {DATA_SUBSET}/train/{animal}')
            os.system(f'cp -r {DATA_COMPLETE}/val/{animal} {DATA_SUBSET}/val/{animal}')
        data = DATA_SUBSET

    model = YOLO(f'storage/yolo/models/yolo11{size}-cls.pt')

    project = "runs_yolo"
    name = f"train_n={n_classes}_size={size}_seed={seed}"
    _ = model.train(
        seed=seed,
        data=data, 
        name=name,
        project=project,
        epochs=100,
    )

    if not n_classes == len(all_animals):
        with open(f'{project}/{name}/selected_animals.json', 'w') as f:
            json.dump(selected_animals, f)
        os.system(f'rm -r {DATA_SUBSET}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', default=0, type=int)
    parser.add_argument('-n', '--n_classes', default=80, type=int)
    parser.add_argument('--size', type=str, required=True)
    args = parser.parse_args()
    launch(args)
