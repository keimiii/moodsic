# Combine features and scores from features_average.csv and the scores csv files in the folder /annotations/annotations averaged per song/song_level/*
# Only need valence_mean, valence_std, arousal_mean, arousal_std from the scores csv files

import pandas as pd
import os

def combine_features_scores():
    # Path to the directory containing the feature CSV files
    features_file = 'features_average.csv'
    scores_dir = 'annotations/annotations averaged per song/song_level'

    # Read the averaged features
    features_df = pd.read_csv(features_file)

    # Initialize an empty DataFrame to hold all scores
    all_scores = pd.DataFrame()

    # Iterate through each file in the scores directory
    for filename in os.listdir(scores_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(scores_dir, filename)
            df = pd.read_csv(file_path, sep=',')
            df['song_id'] = df['song_id'].astype(int)  # Ensure song_id is int
            
            # Select only the relevant columns
            df1 = df[['song_id', ' valence_mean', ' valence_std', ' arousal_mean', ' arousal_std']]
            
            # Concatenate scores into a single DataFrame
            all_scores = pd.concat([all_scores, df1], axis=0)

    # Merge features with scores on 'song_id'
    combined_df = pd.merge(features_df, all_scores, on='song_id', how='inner')
    
    # Shift the columns valence_mean, valence_std, arousal_mean, arousal_std to the start
    score_columns = [' valence_mean', ' valence_std', ' arousal_mean', ' arousal_std']
    other_columns = [col for col in combined_df.columns if (col not in score_columns and col != 'song_id')]
    combined_df = combined_df[['song_id'] + score_columns + other_columns]

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv('combined_features_scores.csv', index=False)
    
def main():
    combine_features_scores()
    print("Combined features and scores saved to 'combined_features_scores.csv'.")
    
if __name__ == "__main__":
    main()