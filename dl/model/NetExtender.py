import torch
import torch.nn as nn


def is_pruned(module: nn.Module) -> bool:
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        return False
    else:
        return True


def traverse_module(module, criterion, layers: list, names: list, prefix="", leaf_only=True):
    if leaf_only:
        for key, submodule in module._modules.items():
            new_prefix = prefix
            if prefix != "":
                new_prefix += '.'
            new_prefix += key
            # is leaf and satisfies criterion
            if len(submodule._modules.keys()) == 0 and criterion(submodule):
                layers.append(submodule)
                names.append(new_prefix)
            traverse_module(submodule, criterion, layers, names, prefix=new_prefix, leaf_only=leaf_only)
    else:
        raise NotImplementedError("Supports only leaf modules")


class Extender:
    DICT_KEY1 = "layers"
    DICT_KEY2 = "layers_prefixes"
    DICT_KEY3 = "relu_layers"
    DICT_KEY4 = "relu_layers_prefixes"
    DICT_KEY5 = "prune_layers"
    DICT_KEY6 = "prune_layers_prefixes"

    def __init__(self, model: nn.Module):
        self.model = model
        self.masks = torch.tensor(0.)
        self.info_dict = self.collect_layers()

    def collect_layers(self) -> dict:
        layers = []
        layers_prefixes = []
        relu_layers = [m for (k, m) in self.model.named_modules() if isinstance(m, nn.ReLU)]
        relu_layers_prefixes = [k for (k, m) in self.model.named_modules() if isinstance(m, nn.ReLU)]
        traverse_module(self.model, lambda x: len(list(x.parameters())) != 0, layers, layers_prefixes)
        prune_indices = [ly_id for ly_id, layer in enumerate(layers) if is_pruned(layer)]
        prune_layers = [layers[ly_id] for ly_id in prune_indices]
        prune_layers_prefixes = [layers_prefixes[ly_id] for ly_id in prune_indices]
        ret = {
            self.DICT_KEY1: layers,
            self.DICT_KEY2: layers_prefixes,
            self.DICT_KEY3: relu_layers,
            self.DICT_KEY4: relu_layers_prefixes,
            self.DICT_KEY5: prune_layers,
            self.DICT_KEY6: prune_layers_prefixes
        }
        return ret
