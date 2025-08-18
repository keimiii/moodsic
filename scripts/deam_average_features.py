# For each csv in /features where the file name is the song_id,
# Output a new csv features_average.csv where the id is song_id and the columns are the global average per feature per song
# Output all the data into a single csv file

import pandas as pd
import os
    
def average_features(features_dir='features'):
    # Initialize an empty DataFrame to hold all features
    all_features = pd.DataFrame()

    # Iterate through each file in the features directory
    for filename in os.listdir(features_dir):
        if filename.endswith('.csv'):
            song_id = filename.split('.')[0]  # Extract song_id from filename
            # Read the CSV file into a DataFrame
            file_path = os.path.join(features_dir, filename)
            df = pd.read_csv(file_path, sep=';')
            df['song_id'] = int(song_id)

            # Ensure the DataFrame has a 'song_id' column
            if 'song_id' in df.columns:
                # Set 'song_id' as index and calculate the mean for each feature
                df.set_index('song_id', inplace=True)
                all_features = pd.concat([all_features, df], axis=0)
            for col in df.columns:
                if col != 'song_id':
                    all_features[col] = pd.to_numeric(all_features[col])

    # Calculate the average for each feature across all songs
    average_features = all_features.groupby(all_features.index).mean()

    # Reset index to have 'song_id' as a column again
    average_features.reset_index(inplace=True)

    # Save the averaged features to a new CSV file
    average_features.to_csv('features_average.csv', index=False)
    
def main():
    average_features()
    print("Averaged features saved to 'features_average.csv'.")
    
if __name__ == "__main__":
    main()