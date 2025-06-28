import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loader import load_and_clean_ipc_text

# Path to the raw sample text file
file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'test_ipc.txt')
file_path = os.path.abspath(file_path)

# Load and clean
cleaned = load_and_clean_ipc_text(file_path)

# Print the first few cleaned lines
for i, line in enumerate(cleaned):
    print(f"{i+1}: {line}")
