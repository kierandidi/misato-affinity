import os
import sys

import plotly.graph_objects as go
import torch as t
from load_md import MdDataset

sys.path.append(os.path.abspath("."))


def create_scatter_data(graph):
    scatter_data = [
        go.Scatter3d(
            x=graph.pos[:, 0],
            y=graph.pos[:, 1],
            z=graph.pos[:, 2],
            mode="markers",
            marker={"size": 10, "color": "black"},
        )
    ]

    for i in range(graph.edge_index.shape[1]):
        edge_type = graph.edge_attr[i][1].long()
        coords = t.stack([graph.pos[graph.edge_index[0][i]], graph.pos[graph.edge_index[1][i]]])

        scatter_data.append(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                marker={"size": 10, "color": "green"},
                line={"width": (1, 3, 5)[edge_type], "color": ("red", "blue", "green")[edge_type]},
            )
        )

    return scatter_data


def main():
    dataset = MdDataset("/data/MD.hdf5")
    d = dataset[0]
    print(d)

    scatter_data = create_scatter_data(d)

    fig = go.Figure(data=scatter_data)
    fig.write_image("fig1.png")


if __name__ == "__main__":
    main()
