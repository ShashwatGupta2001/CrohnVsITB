import matplotlib.pyplot as plt
import json

def plot_results(model_name):
    with open(f"{model_name}_training_log.json", 'r') as f:
        log = json.load(f)
    
    epochs = range(1, len(log['loss']) + 1)

    plt.figure()
    plt.plot(epochs, log['loss'], 'r', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{model_name}_loss.png")

    plt.figure()
    plt.plot(epochs, log['accuracy'], 'b', label='Training accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{model_name}_accuracy.png")

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1]
    plot_results(model_name)
