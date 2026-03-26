import os
import matplotlib.pyplot as plt

def plot_training(history, output_dir="outputs", model_name="model"):
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"C1 {model_name.upper()} Training and Validation Loss Curve")
    plt.savefig(os.path.join(output_dir, f"C1_{model_name}_loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"C1 {model_name.upper()} Training and Validation Accuracy Curve")
    plt.savefig(os.path.join(output_dir, f"C1_{model_name}_accuracy_curve.png"))
    plt.close()