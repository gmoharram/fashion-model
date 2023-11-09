import numpy as np
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    accelerator = "gpu"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    accelerator = "mps"
else:
    device = torch.device("cpu")

from IPython.core.debugger import set_trace


def evaluate_model(model, dataloader):
    report = {}
    accuracy_scores = []

    model.eval()
    model.to(device)
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model.forward(inputs)

        # overall accuracy
        accuracy_scores.append(np.mean((outputs.cpu() == targets.cpu()).numpy()))

    report["accuracy"] = np.mean(accuracy_scores)

    return report
