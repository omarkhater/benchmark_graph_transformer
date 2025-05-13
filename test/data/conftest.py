import pytest
import torch
from torch_geometric.data import Data, Dataset


@pytest.fixture(scope="module")
def one_graph_dataset() -> Dataset:
    """Return a single-graph dataset with boolean masks."""
    class DS(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            x = torch.randn(8, 3)
            y = torch.randint(0, 2, (8,))
            # Add some edges to make clustering possible
            edge_index = torch.tensor(
                [
                    [0, 1, 1, 2, 2, 3, 4, 5, 6, 7],
                    [1, 0, 2, 1, 3, 2, 5, 4, 7, 6]
                ], dtype=torch.long)
            data = Data(x=x, y=y, edge_index=edge_index)
            data.train_mask = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0]).bool()
            data.val_mask = torch.tensor([0, 0, 0, 1, 1, 0, 0, 0]).bool()
            data.test_mask = ~(data.train_mask | data.val_mask)
            return data

    return DS()
