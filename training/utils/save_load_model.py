"""Utils for model saving"""

import os
import pickle


def save_model(model, file_name, directory="models"):
    """Save model as pickle"""
    model = model.cpu()
    model_dict = {"state_dict": model.state_dict(), "hparams": model.hparams}
    if not os.path.exists(directory):
        os.makedirs(directory)
    model_path = os.path.join(directory, file_name)
    pickle.dump(model_dict, open(model_path, "wb", 4))
    return model_path


def load_model(model_class, file_name, directory="models"):
    """Load model from pickle"""
    model_path = os.path.join(directory, file_name)
    model_dict = pickle.load(open(model_path, "rb", 4))
    model = model_class(hparams=model_dict["hparams"])
    model.load_state_dict(model_dict["state_dict"])
    model.eval()
    return model
