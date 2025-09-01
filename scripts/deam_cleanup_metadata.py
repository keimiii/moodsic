import pandas as pd

def clean_2013_metadata():
    def strip_row(row):
        return [item.strip() if isinstance(item, str) else item for item in row]

    # Path to your broken CSV
    input_path = "data/metadata/metadata_2013.csv"
    output_path = "data/metadata/metadata_2013_clean.csv"

    # Read only the first 5 columns (Id, Artist, Song Title, Genre)
    # For each column, strip leading/trailing whitespace
    df = pd.read_csv(input_path, usecols=[0,2,3,6], dtype=str, skipinitialspace=True)

    # Drop rows that are completely empty (if any)
    df = df.dropna(how="all")

    # Apply stripping function to each row
    df = df.apply(strip_row, axis=1, result_type='expand')

    # Save cleaned file
    df.to_csv(output_path, index=False)

    print(f"Cleaned CSV saved to {output_path}")
    
def clean_2014_metadata():
    input_path = "data/metadata/metadata_2014.csv"
    output_path = "data/metadata/metadata_2014_clean.csv"
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        header = next(f)  # skip original header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 5:
                row = [parts[0], parts[1], parts[3], parts[4]]  # Id, Artist, Track, Genre
                rows.append(row)
    
    df = pd.DataFrame(rows, columns=["Id", "Artist", "Track", "Genre"])
    
    # Drop rows that are completely empty (if any)
    df = df.dropna(how="all")

    # Save cleaned file
    df.to_csv(output_path, index=False)

    print(f"Cleaned CSV saved to {output_path}")
    
def clean_2015_metadata():
    input_path = "data/metadata/metadata_2015.csv"
    output_path = "data/metadata/metadata_2015_clean.csv"
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        header = next(f)  # skip original header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6:
                row = [parts[0], parts[2], parts[3], parts[5]]  # Id, Title, Artist, Genre
                rows.append(row)

    df = pd.DataFrame(rows, columns=["Id", "Title", "Artist", "Genre"])
    
    # Drop rows that are completely empty (if any)
    df = df.dropna(how="all")

    # Save cleaned file
    df.to_csv(output_path, index=False)

    print(f"Cleaned CSV saved to {output_path}")

def main():
    clean_2013_metadata()
    print("2013 metadata cleaned.")
    clean_2014_metadata()
    print("2014 metadata cleaned.")
    clean_2015_metadata()
    print("2015 metadata cleaned.")
    
if __name__ == "__main__":
    main()