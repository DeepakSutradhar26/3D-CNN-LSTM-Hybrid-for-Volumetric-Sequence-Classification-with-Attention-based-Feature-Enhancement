import matplotlib.pyplot as plt

def plot_loss_curve(train_loss, val_loss, name):
    plt.figure(figsize=(8,5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Decreasing Over Time")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{name}_loss_curve.png", dpi=300)
    plt.close()

def plot_all_loss(all_train_losses, all_val_losses):
    model_names = ["CNN1", "CNN2", "CNN3"]
    colors = ["blue", "green", "orange"]

    plt.figure(figsize=(10,6))

    for train_losses, val_losses, name, color in zip(all_train_losses, all_val_losses, model_names, colors):
        plt.plot(train_losses, color = color, linewidth=2, label=f"{name} Train")
        plt.plot(val_losses, color=color, linestyle="--",linewidth=2, label=f"{name} Val")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve Comparison (All Models)")
    plt.legend()
    plt.tight_layout()

    plt.savefig("all_models_loss_curve.png", dpi=300)
    plt.close()

def plot_accuracy_curve(acc, name):
    plt.figure(figsize=(10,6))

    plt.plot(acc, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{name}_accuracy.png", dpi=300)
    plt.close()

def plot_all_accuracy(all_acc):
    model_names = ["CNN1", "CNN2", "CNN3"]
    colors = ["blue", "green", "orange"]

    for acc, name, color in zip(all_acc, model_names, colors):
        plt.plot(acc, color=color, linewidth=2, label=f"{name}")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("All models accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig("all_models_accuracy.png", dpi=300)
    plt.close()