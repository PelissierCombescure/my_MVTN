import matplotlib.pyplot as plt

def save_loss_acc(path_to_save, train_losses, test_losses, train_accuracies, test_accuracies):
    # Plotting the loss and accuracy
    plt.figure(figsize=(12, 5))

    # Plot training and testing loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='orange')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and testing accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(test_accuracies, label='Test Accuracy', color='orange')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(path_to_save+'/training_metrics.png')  # Save the figure
    plt.show()  # Display the figure