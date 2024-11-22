import re

def read_and_save_metrics(input_filename, output_filename):
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    
    with open(input_filename, 'r') as file:
        for line in file:
            # Search for each metric using regex, handling missing values
            train_loss_match = re.search(r"loss:\s*([0-9.]+)", line)
            train_accuracy_match = re.search(r"accuracy:\s*([0-9.]+)", line)
            val_loss_match = re.search(r"val_loss:\s*([0-9.]+)", line)
            val_accuracy_match = re.search(r"val_accuracy:\s*([0-9.]+)", line)
            
            # Append values only if they exist
            if train_loss_match:
                train_loss.append(float(train_loss_match.group(1)))
            if train_accuracy_match:
                train_accuracy.append(float(train_accuracy_match.group(1)))
            if val_loss_match:
                val_loss.append(float(val_loss_match.group(1)))
            if val_accuracy_match:
                val_accuracy.append(float(val_accuracy_match.group(1)))
    
    # Write cleaned metrics to the output file
    with open(output_filename, 'w') as outfile:
        outfile.write("Epoch,Train Loss,Train Accuracy,Validation Loss,Validation Accuracy\n")
        for i in range(len(train_loss)):
            outfile.write(f"{i + 1},{train_loss[i]:.4f},{train_accuracy[i]:.4f},"
                          f"{val_loss[i] if i < len(val_loss) else ''},"
                          f"{val_accuracy[i] if i < len(val_accuracy) else ''}\n")
    
    print(f"Metrics saved to {output_filename}")

# Usage
input_file = "functional_model_metrics.txt"  # Your input file
output_file = "functional_model_metrics_cleaned.txt"  # File to save cleaned data
read_and_save_metrics(input_file, output_file)
