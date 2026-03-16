import os
import matplotlib.pyplot as plt

def plot_training(history, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close()

if __name__ == "__main__":
    history = {
        "train_loss": [1.0654, 0.1889, 0.0555, 0.0497, 0.0022, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001],
        "val_loss": [0.3746, 0.2686, 0.2747, 0.2806, 0.2993, 0.2728, 0.2718, 0.2710, 0.2707, 0.2706],
        "train_acc": [0.7197, 0.9508, 0.9874, 0.9851, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "val_acc": [0.8830, 0.9309, 0.9202, 0.9362, 0.9255, 0.9362, 0.9415, 0.9415, 0.9415, 0.9415],
    }

    plot_training(history)
    print("Saved plots to outputs/")