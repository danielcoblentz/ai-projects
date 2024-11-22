import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as font_manager

def read_metrics_from_csv(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)
    
    # Extract the metrics as lists
    train_loss = df['Train Loss'].dropna().tolist()
    train_accuracy = df['Train Accuracy'].dropna().tolist()
    val_loss = df['Validation Loss'].dropna().tolist()
    val_accuracy = df['Validation Accuracy'].dropna().tolist()
    
    # Ensure all lists are the same length by taking the minimum length
    min_length = min(len(train_loss), len(train_accuracy), len(val_loss), len(val_accuracy))
    train_loss = train_loss[:min_length]
    train_accuracy = train_accuracy[:min_length]
    val_loss = val_loss[:min_length]
    val_accuracy = val_accuracy[:min_length]
    
    return train_loss, train_accuracy, val_loss, val_accuracy

def plot_metrics(train_loss, train_accuracy, val_loss, val_accuracy):
    # Professional color palette
    color_palette = {"train": "#3498db", "val": "#e74c3c"}
    
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(11,7), facecolor="#ffffff")
    
    # Line plots
    plt.plot(epochs, train_accuracy, label='Training Accuracy', 
             color=color_palette["train"], linewidth=3.5)
    plt.plot(epochs, val_accuracy, label='Validation Accuracy', 
             color=color_palette["val"], linewidth=3.5, linestyle='--')
    
    # Axes and label styling
    plt.xlabel('Epochs', fontsize=24, labelpad=15, color="#ffffff", fontweight="bold")
    plt.ylabel('Accuracy', fontsize=24, labelpad=15, color="#ffffff", fontweight="bold")
    plt.xticks(fontsize=24, color="#333333", fontweight="bold")
    plt.yticks(fontsize=24, color="#333333", fontweight="bold")
    
    # Axes range
    plt.xlim(0, len(train_loss))
    plt.ylim(0, 1)
    
    # Gridlines
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6, color="#cccccc")
    
    # Remove top and right spines for cleaner look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#aaaaaa')
    plt.gca().spines['left'].set_color('#aaaaaa')
    
    # Legend styling
    font = font_manager.FontProperties(family='Arial', weight='bold', style='normal', size=24)
    plt.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="#e0e0e0", prop=font)
    
    plt.tight_layout()
    output_filename = filename.replace(".txt", ".png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor="#ffffff")
    plt.show()

filename = 'functional_model_metrics_cleaned.txt'
train_loss, train_accuracy, val_loss, val_accuracy = read_metrics_from_csv(filename)
plot_metrics(train_loss, train_accuracy, val_loss, val_accuracy)
