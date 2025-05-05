import numpy as np


class AudioTokenizer:
    """Audio Tokenizer"""

    def __init__(self, unique_gernes) -> None:
        self.__unique_gernes = unique_gernes
        self.__genres2idx = self._make_genres2idx()
        self.__idx2gernes = self._make_idx2genres()

    @property
    def unique_gernes(self):
        """__unique_gernes getter"""

        return self.__unique_gernes

    @property
    def genres2idx(self):
        """__genres2idx getter"""

        if self.__genres2idx is None:
            self.__genres2idx = self._make_genres2idx()

        return self.__genres2idx

    @property
    def idx2gernes(self):
        """__idx2gernes getter"""
        
        if self.__idx2gernes is None:
            self.__idx2gernes = self._make_idx2gernes()
        
        return self.__idx2gernes
            
    def _make_genres2idx(self):
        """Make each gerne to index"""

        genres2idx = {
            genre: idx for idx, genre in enumerate(self.unique_genres)
        }

        return genres2idx
    
    def _make_idx2genres(self):
        """Make index and gerne mapping"""

        idx2genres = {
            idx: genre for genre, idx in self.genres2idx.items()
        }

        return idx2genres
    
    def tokenize(self, genres):
        """Tokenizer for genres"""

        return [
            self.genres2idx[genre]
            for genre in genres if genre in self.genres2idx
        ]

    def detokenize_tolist(self, tokens):
        """Detokenizer for tokens"""

        return [
            self.idx2genres[token] for token in tokens
            if token in self.idx2genres
        ]

    def onehot_encode(self, tokens, max_genres):
        """Onehot encoder for genres"""

        onehot = np.zeros(max_genres)
        onehot[tokens] = 1

        return onehot

    def onehot_decode(self, onehot):
        """Onehot decoder for genres"""

        return [idx for idx, val in enumerate(onehot) if val == 1]


