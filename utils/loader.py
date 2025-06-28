import re

def load_and_clean_ipc_text(file_path):
    """
    Loads and cleans raw IPC text data from the given file.
    Cleaning steps:
      - Remove extra whitespace
      - Remove non-ASCII characters
      - Strip leading/trailing spaces from lines
      - Remove empty lines
    Returns:
      List of cleaned lines.
    """
    cleaned_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Remove non-ASCII characters
            line = line.encode('ascii', errors='ignore').decode()
            # Remove extra whitespace
            line = re.sub(r'\s+', ' ', line)
            # Strip leading/trailing spaces
            line = line.strip()
            if line:
                cleaned_lines.append(line)
    return cleaned_lines