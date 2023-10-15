import os
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from src.data.datasets.md_dataset_invariant import MDDataset


class MDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        graph_path,
        pair_path_train,
        pair_path_test,
        pair_path_val,
        affinity_path,
        batch_size=32,
        num_workers=16,
    ):
        """
        A data module for the MDDataset.

        Parameters:
        - graph_path: Path to the preprocessed graph data.
        - pair_path: Path to the pickle file with pairs data.
        - batch_size: Batch size for the data loaders.
        - num_workers: Number of workers for the data loaders.
        """
        super().__init__()
        assert os.path.exists(graph_path), f"MD data path not found: {graph_path}"
        assert os.path.exists(pair_path_train), f"Pair data path not found: {pair_path_train}"
        assert os.path.exists(pair_path_test), f"Pair data path not found: {pair_path_test}"
        assert os.path.exists(pair_path_val), f"Pair data path not found: {pair_path_val}"

        self.graph_path = graph_path
        self.affinity_path = affinity_path
        self.pair_path_train = pair_path_train
        self.pair_path_test = pair_path_test
        self.pair_path_val = pair_path_val
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """Set up the data module based on the stage (fit/test)."""
        if stage == "fit" or stage is None:
            self.MD_dataset_train = MDDataset(
                self.graph_path, self.pair_path_train, self.affinity_path
            )
            self.MD_dataset_val = MDDataset(self.graph_path, self.pair_path_val, self.affinity_path)

        if stage == "test" or stage is None:
            self.MD_dataset_test = MDDataset(
                self.graph_path, self.pair_path_test, self.affinity_path
            )

    def train_dataloader(self):
        """Returns the train data loader."""
        assert len(self.MD_dataset_train) > 0, "Training set is empty!"
        return DataLoader(
            self.MD_dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        """Returns the validation data loader."""
        assert len(self.MD_dataset_val) > 0, "Validation set is empty!"
        return DataLoader(
            self.MD_dataset_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Returns the test data loader."""
        assert len(self.MD_dataset_test) > 0, "Test set is empty!"
        return DataLoader(
            self.MD_dataset_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
