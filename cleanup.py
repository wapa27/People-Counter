import os
import time
import shutil
from datetime import datetime, timedelta

def delete_old_directories(base_path, pattern="output-", age_seconds=10):
    now = datetime.now()
    for directory in os.listdir(base_path):
        dir_path = os.path.join(base_path, directory)
        
        # Check if it's a directory and matches the pattern
        if os.path.isdir(dir_path) and directory.startswith(pattern):
            # Get the directory's creation or modification time
            dir_time = datetime.fromtimestamp(os.path.getmtime(dir_path))
            
            # Calculate age of the directory
            age = (now - dir_time).total_seconds()
            
            # Delete directory if it's older than the specified age
            if age > age_seconds:
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Error deleting {dir_path}: {e}")

def main():
    base_path = os.getcwd()  # Change to your base path

    while True:
        delete_old_directories(base_path, pattern="output-", age_seconds=10)
        time.sleep(10) 

if __name__ == "__main__":
    main()
