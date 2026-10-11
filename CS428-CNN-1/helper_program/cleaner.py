import re
import csv

def parse_training_log(log_file_path, output_csv_path):
    """
    parses a keras training log and writes the per-epoch metrics to a csv

    parameters:
    - log_file_path: path to the input log file
    - output_csv_path: path to the output csv file
    """
    epoch_regex = re.compile(r'^Epoch\s+(\d+)/(\d+)')
    metrics_regex = re.compile(
        r'accuracy:\s*([0-9.]+)\s*-\s*loss:\s*([0-9.]+)\s*-\s*val_accuracy:\s*([0-9.]+)\s*-\s*val_loss:\s*([0-9.]+)'
    )

    epochs_data = []

    current_epoch = None

    with open(log_file_path, 'r') as file:
        for line in file:
            line = line.strip()

            epoch_match = epoch_regex.match(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            if current_epoch is not None:
                # the metrics line can come a few lines after the header, behind tf warnings
                metrics_match = metrics_regex.search(line)
                if metrics_match:
                    accuracy = float(metrics_match.group(1))
                    loss = float(metrics_match.group(2))
                    val_accuracy = float(metrics_match.group(3))
                    val_loss = float(metrics_match.group(4))

                    epochs_data.append({
                        'Epoch': current_epoch,
                        'Train Loss': loss,
                        'Train Accuracy': accuracy,
                        'Validation Loss': val_loss,
                        'Validation Accuracy': val_accuracy
                    })

                    # reset so a later progress line is not counted for the same epoch
                    current_epoch = None

    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for epoch_data in epochs_data:
            writer.writerow(epoch_data)

    print(f"Extraction complete. Data saved to {output_csv_path}")

# example usage, the paths are from the original training machine
if __name__ == "__main__":
    input_log = '/Users/dancoblentz/Desktop/CS428-CNN-1/terminal_results/new_train-model_code.txt'       # replace with your actual log file path
    output_csv = 'training_metrics44.csv'
    parse_training_log(input_log, output_csv)
