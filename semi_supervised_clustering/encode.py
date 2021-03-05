"""Create encodings of the given data."""
from collections import Counter
from pathlib import Path
from random import getrandbits
from typing import Callable, List, Optional

import numpy as np
import sentencepiece as sp


class Encoder:
    """SentencePiece encoder."""
    
    def __init__(
            self,
            vocab_size: int,
            name: str,
            path: Path,
            clean_f: Callable[..., str] = lambda x: x,
            token_offset: int = 3,
            model_type: str = 'bpe',
    ) -> None:
        """
        Initialise the SentencePiece encoder.

        If the model under the given path exists, it will be loaded. Otherwise, an additional step (create_encoder) is
        required before running the model.

        :param vocab_size: Vocabulary size of the model
        :param name: Name of the model (used to save and load the model)
        :param path: Path under which the model is saved
        :param clean_f: Function used to clean the data before encoding it, only used by model call
        :param token_offset: Number of hidden tokens (3 by default)
        :param model_type: Type of encoding, specific to the SentencePiece package (byte pair encoding by default)
        """
        self.clean_f = clean_f
        self._vocab_size = vocab_size
        self._token_offset = token_offset
        self._name = name
        self._model_type = model_type
        self._path = path
        self._model: Optional[sp.SentencePieceProcessor] = None
        
        # Try to load the model
        self.load()
    
    def __str__(self):
        """Model representation."""
        return f"encoder-{self._name}-{self._model_type}-{self._vocab_size}"
    
    def __repr__(self):
        """Model representation."""
        return str(self)
    
    def __call__(self, sentences: List[str]) -> np.ndarray:
        """Embed a list of sentences, without sampling."""
        return self.encode_batch([self.clean_f(s) for s in sentences], sample=False)
    
    def is_created(self) -> bool:
        """Check if the model has been created."""
        return self._model is not None
    
    def create_encoder(
            self,
            data: List[str],
            show_overview: bool = True,
    ) -> None:
        """Create an encoder best suited for the provided data."""
        # Clean data first
        data = [self.clean_f(d) for d in data]
        
        # Write data to file
        temp_path = Path.cwd() / f'{getrandbits(128)}.txt'
        with open(temp_path, 'w') as file:
            file.write('\n'.join(data))
        
        # Load in file and
        sp.SentencePieceTrainer.train(
                f"--model_type=bpe "
                f"--input={temp_path} "
                f"--model_prefix={self._path / str(self)} "
                f"--vocab_size={self._vocab_size + self._token_offset}"
        )
        
        # Remove temporal file and load the model
        temp_path.unlink(missing_ok=True)
        assert self.load()
        
        # Display overview
        if show_overview:
            # Get 5 most occurring words
            most_occurring = [k for k, _ in sorted(Counter(data).items(), key=lambda x: -x[1])][:5]
            self.show_overview(words=most_occurring)
    
    def encode(self, sentence: str, sample: bool = False) -> np.ndarray:
        """
        Encode a single sentence.

        :param sentence: Sentence to encode, assumed that it has already been cleaned
        :param sample: Sample the encoding, potentially representing the sentence by different tokens
        """
        assert self._model is not None
        encoding = self._model.sample_encode_as_ids(sentence, -1, 0.1) if sample else self._model.encode(sentence)
        vec = np.zeros((self._vocab_size,), dtype='float32')
        for enc in encoding:
            vec[enc - self._token_offset] += 1.
        return vec
    
    def encode_batch(self, sentences: List[str], sample: bool = False) -> np.ndarray:
        """
        Encode a batch of sentences.

        :param sentences: List of sentences to encode, assumed that it has already been cleaned
        :param sample: Sample the encoding, potentially representing the sentence by different tokens
        """
        arr = np.zeros((len(sentences), self._vocab_size), dtype='float32')
        for i, s in enumerate(sentences):
            arr[i] = self.encode(s, sample=sample)
        return arr
    
    def show_overview(self, words: Optional[List[str]]) -> None:
        """Analyse the current model."""
        print(f"\n\nAnalysing encoder '{self}'")
        
        # Analyse model
        for word in words:
            print(f"\nAnalysing word '{word}':")
            enc = self._model.encode(self.clean_f(word))
            print(f" --> Encoding: {enc}")
            print(" --> Word by word:")
            for e in enc:
                print(f"     - {e}: {self._model.decode(e)}")
        
        # Check all letter of the alphabet
        print("\nEncoding the alphabet:")
        for x in "abcdefghijklmnopqrstuvwxyz":
            enc = self._model.encode(x)
            print(f" - {x} = {enc} = {self._model.decode(enc)}")
        
        # Analyse positions in the model
        print("\nAnalyse the model's vocabulary:")
        for i in range(self._token_offset, self._vocab_size + self._token_offset, self._vocab_size // 10):
            print(f" - Index {i:3d}: {self._model.decode(i)}")
        print(f"\n\n")
    
    def load(self) -> bool:
        """Try to load a pretrained model and return its success."""
        if (self._path / f"{self}.model").is_file():
            self._model = sp.SentencePieceProcessor(model_file=str(self._path / f"{self}.model"))
            return True
        return False
