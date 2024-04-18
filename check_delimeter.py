import csv

def detect_delimiter(csv_file, sample_size=5):
    delimiters = [',', ';', '\t']  # List of common delimiters to check

    with open(csv_file, 'r', newline='') as file:
        # Read the first few lines of the CSV file
        sample_data = [file.readline() for _ in range(sample_size)]

    # Count occurrences of each delimiter in the sample data
    delimiter_counts = {delimiter: sum(line.count(delimiter) for line in sample_data) for delimiter in delimiters}

    # Determine the most frequent delimiter
    most_frequent_delimiter = max(delimiter_counts, key=delimiter_counts.get)

    return most_frequent_delimiter

# Example usage
csv_file_path = 'iphi2802.csv'
detected_delimiter = detect_delimiter(csv_file_path)
print("Detected delimiter:", repr(detected_delimiter))
