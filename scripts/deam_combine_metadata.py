# This script is to combine all metadata rows in the folder /data/metadata into a single csv file, /data/all_metadata.csv
import pandas as pd
import os

def combine_metadata():
    metadata_dir = 'data/metadata'
    all_metadata = pd.DataFrame()
    for filename in os.listdir(metadata_dir):
        if filename.endswith('clean.csv'):
            file_path = os.path.join(metadata_dir, filename)
            print(file_path)
            df = pd.read_csv(file_path, sep=',')
            all_metadata = pd.concat([all_metadata, df], axis=0)
    # Sort df by Id column
    all_metadata = all_metadata.sort_values(by='Id')
    # Save to all_metadata.csv
    all_metadata.to_csv('data/all_metadata.csv', index=False)
    
def main():
    combine_metadata()
    print("Combined metadata saved to 'all_metadata.csv'.")
    
if __name__ == "__main__":
    main()