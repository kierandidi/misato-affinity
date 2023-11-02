import torch
from pytorch_lightning import LightningModule
from torch.nn import BatchNorm1d, Linear, ReLU
from torch.nn import functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import global_max_pool, global_mean_pool


class GraphConvNetwork(LightningModule):
    """
    Graph Convolutional Network model class based on GCNConv.
    """

    def __init__(self, in_dim, out_dim=1, hidden_dim=64, lr=1e-6):
        """
        Initialization for GCN model.

        input_dim: int - The dimensionality of input node features.
        hidden_dim1: int - The number of hidden units in the first GCN layer.
        hidden_dim2: int - The number of hidden units in the second GCN layer.
        lr: float - learning rate for the optimizer.
        """
        super(GraphConvNetwork, self).__init__()

        # Graph Convolution (GCN) layers
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.bn2 = BatchNorm1d(hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim * 4)
        self.bn3 = BatchNorm1d(hidden_dim * 4)
        self.conv4 = GCNConv(hidden_dim * 4, hidden_dim * 4)
        self.bn4 = BatchNorm1d(hidden_dim * 4)
        self.conv5 = GCNConv(hidden_dim * 4, hidden_dim * 2)
        self.bn5 = BatchNorm1d(hidden_dim * 2)

        # Output, 3 linear layers with ReLU activation
        self.fc1 = Linear(hidden_dim * 8, hidden_dim * 4)
        self.fc2 = Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = Linear(hidden_dim, out_dim)

        # Non-linear activation
        self.relu = ReLU()

        # Learning rate
        self.lr = lr

    def forward_one_complex(self, batch):
        """
        Forward pass for one of the complexes the GCN model.

        batch: Batch - The batch containing all the necessary attributes.
        """
        x, edge_index, edge_weight = batch.x, batch.edge_index, batch.edge_attr[:, 0]
        indices = x[:, -1].clone()
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.conv4(x, edge_index, edge_weight)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x, edge_index, edge_weight)
        x = self.bn5(x)
        # pool separately for protein and ligand, where x[:,-1]==1 is protein and x[:,-1]==0 is ligand
        x_prot = global_mean_pool(x[indices == 0], batch=batch.batch[indices == 0])
        x_lig = global_mean_pool(x[indices == 1], batch=batch.batch[indices == 1])

        return x_prot, x_lig

    def forward(self, batch1, batch2):
        """
        Forward pass for the GCN model.

        batch: Batch - The batch containing all the necessary attributes.
        """

        x_prot1, x_lig1 = self.forward_one_complex(batch1)
        x_prot2, x_lig2 = self.forward_one_complex(batch2)

        out = torch.cat([x_prot1, x_prot2, x_lig1, x_lig2], dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        batch1, batch2 = batch
        x_1, edge_index_1, edge_weight_1, y_1 = (
            batch1.x,
            batch1.edge_index,
            batch1.edge_attr[:, 0],
            batch1.y,
        )
        x_2, edge_index_2, edge_weight_2, y_2 = (
            batch2.x,
            batch2.edge_index,
            batch2.edge_attr[:, 0],
            batch2.y,
        )

        y_hat = self.forward(batch1, batch2)
        y_hat = y_hat.view(-1)

        y = torch.log(y_1 / y_2)

        loss = F.mse_loss(y_hat, y)

        self.log("y_hat", y_hat[0], batch_size=y.shape[0])
        self.log("y", y[0], batch_size=y.shape[0])
        self.log("train_loss", loss, batch_size=y.shape[0])

        # print(f"train_loss: {loss}")
        return loss

    def validation_step(self, batch, batch_idx):
        batch1, batch2 = batch
        x_1, edge_index_1, edge_weight_1, y_1 = (
            batch1.x,
            batch1.edge_index,
            batch1.edge_attr[:, 0],
            batch1.y,
        )
        x_2, edge_index_2, edge_weight_2, y_2 = (
            batch2.x,
            batch2.edge_index,
            batch2.edge_attr[:, 0],
            batch2.y,
        )

        y_hat = self.forward(batch1, batch2)
        y_hat = y_hat.view(-1)


        y = torch.log(y_1 / y_2)
        loss = F.mse_loss(y_hat, y)

        self.log("val_loss", loss, batch_size=y.shape[0])
        return loss

    def predict_step(self, batch, batch_idx):
        batch1, batch2 = batch
        x_1, edge_index_1, edge_weight_1, y_1 = (
            batch1.x,
            batch1.edge_index,
            batch1.edge_attr[:, 0],
            batch1.y,
        )
        x_2, edge_index_2, edge_weight_2, y_2 = (
            batch2.x,
            batch2.edge_index,
            batch2.edge_attr[:, 0],
            batch2.y,
        )

        y_hat = self.forward(batch1, batch2)
        y_hat = y_hat.view(-1)

        y = torch.log(y_1 / y_2)
        return (y, y_hat, (batch1.pdb_id, batch2.pdb_id))
