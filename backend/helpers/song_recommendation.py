def recommend_song(valence, arousal, cluster_id):
    """
    Recommend a song based on valence, arousal, and cluster
    """
    # Use static mock data for now
    mock_songs = [
        {"song_id": "10", "title": "Sunlit Avenues", "artist": "Ivory Coast", "genre": "electronic"},
        {"song_id": "1000", "title": "Gravity Rush", "artist": "Midnight Circuit", "genre": "rock"},
        {"song_id": "1001", "title": "Midnight Drizzle", "artist": "Slow Parade", "genre": "folk/country"},
        {"song_id": "1002", "title": "Caps & Gowns", "artist": "Riverfolk", "genre": "pop"},
        {"song_id": "1003", "title": "Full Court Press", "artist": "Baseline", "genre": "hip-hop"},
        {"song_id": "1004", "title": "Couch Screams", "artist": "Neon Noir", "genre": "electronic"}
    ]
    return mock_songs[cluster_id % len(mock_songs)]