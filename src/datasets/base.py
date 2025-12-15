import os
import re
import logging
import json
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()

        # ========== Initialization ==========

        # Verify data path
        abs_path = os.path.abspath(data_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Data file not found.\nExpected path: {abs_path}\n")

        logger.info(f"Loading data from: {abs_path}")

        # Define column names
        col_names = {
            'ratings': ['user_id', 'movie_id', 'rating', 'timestamp'],
            'users': ['user_id', 'gender', 'age', 'occupation', 'zipcode'],
            'movies': ['movie_id', 'title', 'genres']
        }

        # Load data
        ratings = pd.read_csv(
            os.path.join(abs_path, 'ratings.dat'),
            sep='::', header=None, engine='python', names=col_names['ratings'],
            encoding='latin-1'
        )
        users = pd.read_csv(
            os.path.join(abs_path, 'users.dat'),
            sep='::', header=None, engine='python', names=col_names['users'],
            encoding='latin-1'
        )
        movies = pd.read_csv(
            os.path.join(abs_path, 'movies.dat'),
            sep='::', header=None, engine='python', names=col_names['movies'],
            encoding='latin-1'
        )

        # ========== Preprocessing ==========

        # Adjust IDs for PyTorch Embedding (0-based)
        ratings['user_id'] -= 1
        ratings['movie_id'] -= 1
        users['user_id'] -= 1
        movies['movie_id'] -= 1

        # Calculate num_users/num_movies
        self.num_users = int(ratings['user_id'].max()) + 1
        self.num_items = int(ratings['movie_id'].max()) + 1

        # Get release year in movie title
        movies['release_year'] = movies['title'].apply(lambda x: re.search(r'\((\d{4})\)', x).group(1))

        # Get movie genres (FIRST genre only)
        # !!! CAUTION !!! Multi-hot encoding can be implemented here for advanced models
        movies['genres'] = movies['genres'].apply(lambda x: x.split('|')[0])

        # Merge all data (left join)
        df = pd.merge(ratings, users, on='user_id', how='left')
        df = pd.merge(df, movies, on='movie_id', how='left')

        # Convert time_stamp into datetime
        dt = pd.to_datetime(df['timestamp'], unit='s')
        df['year'] = dt.dt.year
        df['month'] = dt.dt.month
        df['day'] = dt.dt.day
        df['weekday'] = dt.dt.weekday
        df['hour'] = dt.dt.hour

        # Find user's last rated movie
        df = df.sort_values(by=['user_id', 'timestamp'])
        df['last_item_id'] = df.groupby('user_id')['movie_id'].shift(1)
        unknown_item_token = self.num_items # assign num_items to represent 'No previous item' (Cold Start)
        df['last_item_id'] = df['last_item_id'].fillna(unknown_item_token).astype(int)

        # Store user's history rated movie
        df = df.sort_values(by=['user_id', 'timestamp'])
        df['history_len'] = df.groupby('user_id').cumcount() # total history length BEFORE current timestamp

        self.user_history_dict = df.groupby('user_id')['movie_id'].apply(list).to_dict()
        self.max_history_len = max([len(seq) for seq in self.user_history_dict.values()])
        self.padding_token = self.num_items # assign num_items to represent 'No history item' (Cold Start)

        # ========== Feature encoding ==========

        self.feature_dims = {}
        sparse_features = [
            'gender', 'age', 'occupation', 'zipcode', 'genres',
            'release_year',
            'year', 'month', 'day', 'weekday', 'hour',
            #'last_item_id',    # Excluded to maintain ID alignment with 'item_id' (Shared Embedding)
                                # Reason: LabelEncoder would change last_item_id
        ]
        for feat in sparse_features:
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])
            self.feature_dims[feat] = df[feat].nunique()

        # Manually add other fields
        self.feature_dims['user_id'] = self.num_users
        self.feature_dims['item_id'] = self.num_items + 1  # Indices: [0, N-1] for items, [N] for cold-start
        self.feature_dims['last_item_id'] = self.num_items + 1 # Indices: [0, N-1] for items, [N] for cold-start

        # ========== Finalization ==========

        # rename: movie_id -> item_id
        df.rename(columns={'movie_id': 'item_id'}, inplace=True)

        self.df = df

        self.feature_names = list(self.feature_dims.keys())

        # ========== Log ==========
        logger.info(" Dataset Info ".center(60,"="))
        logger.info(f"Total Samples: {len(self.df)}")
        logger.info(f"Total Features: {len(self.feature_dims)}")
        #logger.info(f"Feature Names : {feature_names}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raise NotImplementedError("BaseDataset should not be used directly. Use subclasses.")
