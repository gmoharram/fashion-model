import matplotlib.pyplot as plt
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


def visualizer(model, test_data=None, num_imgs=1):
    model.eval()
    model.to(device)
    num_example_imgs = num_imgs
    plt.figure(figsize=(15, 5 * num_example_imgs))
    for i, (img, target) in enumerate(test_data[:num_example_imgs]):
        output = model.forward(img.unsqueeze(0).to(device))

        output = output.squeeze(0).cpu()
        img, target, output = img.numpy(), target.numpy(), output.detach().numpy()

        # img
        plt.subplot(num_example_imgs, 3, i * 3 + 1)
        plt.axis("off")
        plt.imshow(img.transpose(1, 2, 0))
        if i == 0:
            plt.title("Input image")

        # target
        plt.subplot(num_example_imgs, 3, i * 3 + 2)
        plt.axis("off")
        plt.imshow(target.transpose(1, 2, 0))
        if i == 0:
            plt.title("Target image")

        # pred
        plt.subplot(num_example_imgs, 3, i * 3 + 3)
        plt.axis("off")
        plt.imshow(output.transpose(1, 2, 0))
        if i == 0:
            plt.title("Prediction image")

    plt.show()


def visualize_dataset(dataset, num_imgs=1):
    num_example_imgs = num_imgs
    plt.figure(figsize=(15, 5 * num_example_imgs))
    for i, (img, target) in enumerate(dataset[:num_example_imgs]):
        img, target = img.numpy(), target.numpy()

        # img
        plt.subplot(num_example_imgs, 2, i * 2 + 1)
        plt.axis("off")
        plt.imshow(img.transpose(1, 2, 0))
        if i == 0:
            plt.title("Input image")

        # target
        plt.subplot(num_example_imgs, 2, i * 2 + 2)
        plt.axis("off")
        plt.imshow(target.transpose(1, 2, 0))
        if i == 0:
            plt.title("Target image")

    plt.show()
