import torch


def export_model(path, model):
    torch.save(model.state_dict(), path)
