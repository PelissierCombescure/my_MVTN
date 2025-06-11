import matplotlib.pyplot as plt

def save_loss_acc(path_to_save, train_losses, test_losses, train_accuracies, test_accuracies, plot_best = False, best_epoch = None):
    
    mess_attention = "Dans json epoch commence à 1, \ndonc ici best_epoch = best_epoch_json - 1"
    # Plotting the loss and accuracy
    plt.figure(figsize=(12, 5))

    # Plot training and testing loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='orange')
    plt.title('Loss Over Epochs\n'+mess_attention)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Best epoch
    if plot_best:
        plt.scatter(x=best_epoch-1, y=test_losses[best_epoch-1], color='red', linestyle='--', label='Best Epoch')
        plt.scatter(x=best_epoch-1, y=train_losses[best_epoch-1], color='red', linestyle='--')
        plt.text(best_epoch-1, test_losses[best_epoch-1], f'({best_epoch-1}, {test_losses[best_epoch-1]:.2f})', fontsize=9, color='red')
        plt.text(best_epoch-1, train_losses[best_epoch-1], f'({best_epoch-1}, {train_losses[best_epoch-1]:.2f})', fontsize=9, color='red')
        plt.legend()

    # Plot training and testing accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(test_accuracies, label='Test Accuracy', color='orange')
    plt.title('Accuracy Over Epochs\n'+mess_attention)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%) ')
    plt.legend()
    
        # Best epoch
    if plot_best:
        plt.scatter(x=best_epoch-1, y=test_accuracies[best_epoch-1], color='red', linestyle='--', label='Best Epoch')
        plt.scatter(x=best_epoch-1, y=train_accuracies[best_epoch-1], color='red', linestyle='--')
        plt.text(best_epoch-1, test_accuracies[best_epoch-1], f'({best_epoch-1}, {test_accuracies[best_epoch-1]:.2f})', fontsize=9, color='red')
        plt.text(best_epoch-1, train_accuracies[best_epoch-1], f'({best_epoch-1}, {train_accuracies[best_epoch-1]:.2f})', fontsize=9, color='red')
        plt.legend()

    plt.tight_layout()
    plt.savefig(path_to_save+'/training_metrics.png')  # Save the figure
    plt.show()  # Display the figure