from torch_geometric.data import Data


class ExampleTransform(object):
    def __init__(self, cfg):
        pass

    def __call__(self, data: Data) -> Data:
        ## Do some transformation here

        return data


class ToPyGData(object):
    def __init__(self, cfg):
        pass

    def __call__(self, data: dict) -> Data:
        new_data = Data(**data)

        return new_data


def get_transforms(cfg):
    transforms = []

    for tranform in cfg.transforms.keys():
        if tranform == "example_transofrosm":
            transforms.append(ExampleTransform(cfg))

    return transforms
