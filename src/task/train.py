import os
import pathlib as pl

import pandas as pd
from torch.utils.data import DataLoader

import wandb

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from src.pipeline import train, test

from src.model import ModelSelector
from src.optimization import OptimizationSelector
from src.loss import LossSelector
from src.data import DatasetSelector

from src.logger import logger

class TrainTask:
    def __init__(self, model_selector, optimizer_selector, loss_selector, dataset_selector, epochs, batch_size, k_folds, save_path):
        self.model_selector = model_selector
        self.optimizer_selector = optimizer_selector
        self.loss_selector = loss_selector
        self.dataset_selector = dataset_selector
        self.epochs = epochs
        self.batch_size = batch_size
        self.k_folds = k_folds
        self.save_path = save_path
        self.save_results_path = None

    def _make_save_dir(self):
        count = 1
        while True:
            save_dir = f"{self.model_selector.name}_{self.optimizer_selector.name}_{self.dataset_selector.name}_{count}"
            save_dir_path = pl.Path(self.save_path) / pl.Path(save_dir)
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)
                logger.info(f"Created directory to {save_dir_path}")
                break
            count += 1
        
        self.save_results_path = save_dir_path

    def _make_fold_save_dir(self, fold):
        os.makedirs(self.save_results_path / f"fold_{fold}")
        return self.save_results_path / f"fold_{fold}"

    def run(self, train_loader, val_loader, epochs):
        model = self.model_selector.get_model()
        optimizer = self.optimizer_selector.get_optimizer(model)
        loss = self.loss_selector.get_loss()
        dataset = self.dataset_selector.get_dataset()

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            logger.info(f"Created root directory to save all results {self.save_path}")

        # make saving dir with format model_name_optimizer_name_dataset_day_month_year
        save_dir = f"{model.name}_{self.optimizer_selector.name}_{self.dataset_selector.name}"
        save_dir_path = pl.Path(self.save_path) / pl.Path(save_dir)
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
            self.save_results_path = save_dir_path
            logger.info(f"Created directory to {save_dir_path}")
        elif os.path.exists(save_dir_path):
            logger.info(f"Directory {save_dir_path} already exists")
            self._make_save_dir()

        # save fold results
        folds_paths_df = dataset.folds_df
        folds_paths_df.to_csv(self.save_results_path / "folds_paths.csv", index=False)

        test_dataset = dataset.test_dataset
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        for fold in self.k_folds:
            train_dataset, val_dataset = self.dataset_selector.get_k_fold_train_val_tuple(fold)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

            fold_dir = str(self._make_fold_save_dir(fold))
            train_losses, train_accs, vals_losses, vals_accs = train(model, optimizer, loss, train_loader, val_loader, epochs, fold_dir)
            
            # save fold results
            fold_results = {
                "train_losses": train_losses,
                "train_accs": train_accs,
                "val_losses": vals_losses,
                "val_accs": vals_accs
            }

            fold_results_df = pd.DataFrame(fold_results)
            fold_results_df.to_csv(fold_dir / "fold_results.csv", index=False)

            # test model
            test_loss, _, y_pred, y_true = test(model, loss, test_loader)

            recall_score_val = recall_score(y_true, y_pred)
            precision_score_val = precision_score(y_true, y_pred)
            f1_score_val = f1_score(y_true, y_pred)
            accuracy_score_val = accuracy_score(y_true, y_pred)

            wandb.log({f"test/loss/fold{fold}": test_loss, f"test/recall/fold{fold}": recall_score_val, f"test/precision/fold{fold}": precision_score_val, f"test/f1/fold{fold}": f1_score_val, f"test/accuracy/fold{fold}": accuracy_score_val})

            results = {
                "test_loss": test_loss,
                "recall_score": recall_score_val,
                "precision_score": precision_score_val,
                "f1_score": f1_score_val,
                "accuracy_score": accuracy_score_val
            }

            # save metrics
            results_df = pd.DataFrame(results, index=[0])
            results_df.to_csv(fold_dir / "test_results.csv", index=False)

            preds_true = {
                "preds": y_pred,
                "true": y_true
            }
            
            # save predictions
            preds_true_df = pd.DataFrame(preds_true)
            preds_true_df.to_csv(fold_dir / "preds_true.csv", index=False)