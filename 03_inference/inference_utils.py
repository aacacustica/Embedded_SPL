import os



def filter_predictions(predictions,threshold):
    import numpy as np
    top_i = int(np.argmax(predictions))
    top_pred = float(predictions[top_i])
    if top_pred > threshold:
        return top_i, top_pred
    return None, None


def load_processed_files(processed_file_path):
    """Load the set of processed filenames from a text file."""
    if os.path.exists(processed_file_path):
        with open(processed_file_path, "r") as f:
            return {line.strip() for line in f if line.strip()}
    return set()

def update_processed_files(processed_file_path, filename):
    """Append a processed filename to the text file."""
    with open(processed_file_path, "a") as f:
        f.write(filename + "\n")