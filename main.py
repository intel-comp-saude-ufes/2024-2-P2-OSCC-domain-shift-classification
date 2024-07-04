import pathlib as pl
import os

import click
from click_params import DecimalRange

from sklearn.utils.class_weight import compute_class_weight

from src.logger import logger
from src.task.train import TrainTask
from src.model import ModelSelector
from src.optimization import OptimizationSelector
from src.loss import LossSelector
from src.data import DatasetSelector

PATCH_PATH = pl.Path(os.path.abspath("")) / pl.Path('data') / pl.Path('ndb_ufes') / pl.Path('patches')
IMAGE_PATH = pl.Path(os.path.abspath("")) / pl.Path('data') / pl.Path('ndb_ufes') / pl.Path('images') 
METADATA_PATH = pl.Path(os.path.abspath("")) / pl.Path('data') / pl.Path('ndb_ufes') / pl.Path('ndb-ufes.csv') 

@click.command()
@click.option("--train", is_flag=True, help="Train flag. When used along with test prevails over test flag", default=False)
@click.option("--test", is_flag=True, help="Test flag. When used along with train, train prevails.", default=False)
@click.option("--optimizer_name", default="sgd", help="Optimizer name", type=click.Choice(["adam", "sgd"]), show_default=True)
@click.option("--learning_rate", default=0.001, help="Learning rate", type=float, show_default=True)
@click.option("--scheduler_name", default="reduce_lr_on_plateau", help="Scheduler name", type=click.Choice(["step_lr", "reduce_lr_on_plateau"]), show_default=True)
@click.option("--step_size", default=30, help="Step size for StepLR scheduler", type=int, show_default=True)
@click.option("--gamma", default=0.1, help="Gamma for StepLR scheduler", type=float, show_default=True)
@click.option("--mode", default="min", help="Mode for    scheduler", type=str, show_default=True)
@click.option("--factor", default=0.1, help="Factor for ReduceLROnPlateau scheduler", type=float, show_default=True)
@click.option("--patience", default=10, help="Patience for ReduceLROnPlateau scheduler", type=int, show_default=True)
@click.option("--min_lr", default=10e-6, help="Min lr for ReduceLROnPlateau scheduler", type=float, show_default=True)
@click.option("--loss_name", default="cross_entropy", help="Loss name", type=click.Choice(["cross_entropy"]), show_default=True)
@click.option("--use_weights_loss", default=True, help="Use weights", type=bool, show_default=True)
@click.option("--dataset_name", default="patches_ndb", help="Dataset name", type=click.Choice(["patches_ndb", "rahman"]), show_default=True)
@click.option("--dataset_path", default=PATCH_PATH, help="Dataset path", type=str, show_default=True)
@click.option("--train_size", default=0.8, help="Train size", show_default=True, type=DecimalRange(0, 1))
@click.option("--k_folds", default=5, help="K folds", type=int, show_default=True)
@click.option("--model_name", default="densenet121", help="Model name", type=click.Choice(["densenet121"]), show_default=True)
@click.option("--num_classes", default=2, help="Number of classes", type=int, show_default=True)
@click.option("--model_weights_path", default='default', help="Model weights path", type=str, show_default=True)
@click.option("--epochs", default=200, help="Number of epochs", type=int, show_default=True)
@click.option("--batch_size", default=32, help="Batch size", type=int, show_default=True)
@click.option("--save_path", default="results", help="Save path", type=str, show_default=True)
def main(train, test, optimizer_name, learning_rate, scheduler_name, step_size, gamma, mode, factor, patience, min_lr, loss_name, \
         use_weights_loss, dataset_name, dataset_path, train_size, k_folds, model_name, num_classes, model_weights_path, epochs,  \
         batch_size, save_path):    
    # convert to Path
    dataset_path = pl.Path(dataset_path)
    save_path = pl.Path(save_path)
    model_weights_path = pl.Path(model_weights_path) if model_weights_path.lower() != "default" else model_weights_path

    train_size = float(train_size)
    dataset_selector = DatasetSelector(dataset_name, dataset_path, train_size=train_size, k_folds=k_folds)
    dataset = dataset_selector.get_dataset()
    num_classes = len(dataset.labels_names)

    # initialize model
    print(model_name, num_classes, model_weights_path)
    model_selector = ModelSelector(model_name, model_weights_path, num_classes=num_classes)
    model = model_selector.get_model()

    if train:
        logger.info(f"Training model {model_name} with dataset {dataset_name}")
        optimization_selector = OptimizationSelector(optimizer_name, scheduler_name, learning_rate, step_size=step_size, gamma=gamma, mode=mode, factor=factor, patience=patience, min_lr=min_lr)
        optimizer = optimization_selector.get_optimizer(model.parameters())
        scheduler = optimization_selector.get_scheduler()

        if use_weights_loss:
            class_weights = compute_class_weight('balanced', classes=dataset.labels_names.values(), y=dataset.labels)

        loss_selector = LossSelector(loss_name, class_weights)

        train_task = TrainTask(model, optimizer, scheduler, loss_selector, dataset, epochs, batch_size, k_folds=k_folds, save_path=save_path)
        train_task.run()
    elif test:
        print("testing")
    else:
        print("no action")

if __name__ == "__main__":
    main()