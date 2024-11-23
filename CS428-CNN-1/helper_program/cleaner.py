import re
import csv

def parse_training_log(log_file_path, output_csv_path):
    """
    Parses a Keras training log file and extracts epoch metrics into a CSV file.

    Parameters:
    - log_file_path: Path to the input log file.
    - output_csv_path: Path to the output CSV file.
    """
    # Regular expressions to match epoch lines and metrics
    epoch_regex = re.compile(r'^Epoch\s+(\d+)/(\d+)')
    metrics_regex = re.compile(
        r'accuracy:\s*([0-9.]+)\s*-\s*loss:\s*([0-9.]+)\s*-\s*val_accuracy:\s*([0-9.]+)\s*-\s*val_loss:\s*([0-9.]+)'
    )

    # List to hold all extracted epoch data
    epochs_data = []

    current_epoch = None

    with open(log_file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Check for epoch start
            epoch_match = epoch_regex.match(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue  # Move to the next line to find metrics

            if current_epoch is not None:
                # Try to find metrics in the current line
                metrics_match = metrics_regex.search(line)
                if metrics_match:
                    accuracy = float(metrics_match.group(1))
                    loss = float(metrics_match.group(2))
                    val_accuracy = float(metrics_match.group(3))
                    val_loss = float(metrics_match.group(4))

                    # Append the extracted data
                    epochs_data.append({
                        'Epoch': current_epoch,
                        'Train Loss': loss,
                        'Train Accuracy': accuracy,
                        'Validation Loss': val_loss,
                        'Validation Accuracy': val_accuracy
                    })

                    # Reset current_epoch to avoid duplicate entries
                    current_epoch = None

    # Write the extracted data to a CSV file
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for epoch_data in epochs_data:
            writer.writerow(epoch_data)

    print(f"Extraction complete. Data saved to {output_csv_path}")

# Example usage:
if __name__ == "__main__":
    input_log = '/Users/dancoblentz/Desktop/CS428-CNN-1/terminal_results/new_train-model_code.txt'       # Replace with your actual log file path
    output_csv = 'training_metrics44.csv'  # Desired output CSV file path
    parse_training_log(input_log, output_csv)
