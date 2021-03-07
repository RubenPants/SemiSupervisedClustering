"""Create the embedding model."""
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import tensorflow as tf


class Embedder:
    """Custom embedding model."""
    
    def __init__(
            self,
            name: str,
            input_size: int,
            layers: List[int],
            path: Path,
            normalise: bool = True,
    ) -> None:
        """
        Initialise the embedding model.

        :param name: Name of the model to load
        :param input_size: Input size of the model
        :param layers: Dense layers of the model, in chronological order (last layer is the embedding space)
        :param path: Path under which the model is saved
        :param normalise: Whether or not to normalise the model's outputs
        """
        assert len(layers) >= 1
        self._name = name
        self._input_size = input_size
        self._layers = layers
        self._normalise = normalise
        self._path = path
        self._model: Optional[tf.keras.Model] = None
        
        # Try to load the model, create new if unsuccessful
        if not self.load():
            self.create_model()
    
    def __str__(self):
        """Model representation."""
        return (
            f"embedder-"
            f"{self._name}-"
            f"{self._input_size}-"
            f"{'-'.join([f'{l}' for l in self._layers])}"
            f"{'-normalised' if self._normalise else ''}"
        )
    
    def __repr__(self):
        """Model representation."""
        return str(self)
    
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Embed the given input."""
        return self._model(inputs).numpy()
    
    def create_model(
            self,
            show_overview: bool = True,
    ) -> None:
        """Create a new model under the requested specifications."""
        print(f"Creating new model...")
        inputs = tf.keras.Input(
                shape=(self._input_size,),
                name='Input',
        )
        
        # Intermediate layers
        x = inputs
        for i, layer_size in enumerate(self._layers[:-1]):
            x = tf.keras.layers.Dense(
                    layer_size,
                    name=f'Dense-{i}',
            )(x)
        
        # Output layer (embedding)
        x = tf.keras.layers.Dense(
                self._layers[-1],
                activation='tanh',
                name='Dense-output',
        )(x)
        
        # Normalise the output layer, if requested
        if self._normalise:
            x = tf.keras.layers.LayerNormalization(
                    axis=1,
                    name="Normalization",
            )(x)
        
        # Create the model
        self._model = tf.keras.Model(
                inputs=inputs,
                outputs=x,
                name=str(self),
        )
        self._model.compile(optimizer='adam')
        if show_overview:
            self._model.summary()
        self.store()
    
    def train_positive(
            self,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int = 1024,
    ) -> float:
        """Train a positive-sampling sequence for the model and return the corresponding loss."""
        self._model.compile(optimizer='adam', loss=self._positive_loss)
        return self._model.fit(x, y, batch_size=batch_size, verbose=0).history['loss'][0]
    
    def train_negative(
            self,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int = 1024,
    ) -> float:
        """Train a negative-sampling sequence for the model and return the corresponding loss."""
        self._model.compile(optimizer='adam', loss=self._negative_loss)
        return self._model.fit(x, y, batch_size=batch_size, verbose=0).history['loss'][0]
    
    def store(self) -> None:
        """Store the current model state."""
        self._model.save(self._path / str(self))
    
    def load(self) -> bool:
        """Try to load a pretrained model and return its success."""
        if (self._path / str(self)).is_dir():
            self._model = tf.keras.models.load_model(
                    self._path / str(self),
                    custom_objects={
                        '_positive_loss': self._positive_loss,
                        '_negative_loss': self._negative_loss,
                    },
            )
            return True
        return False
    
    def _positive_loss(self, y_true, y_pred) -> Any:
        """MAE loss on the dot-difference for positive sampling (https://www.desmos.com/calculator/uh9yklg2ij)."""
        y_true = tf.reshape(y_true, (-1, self._layers[-1]), name='reshape_y_true')
        y_true = tf.math.l2_normalize(y_true, axis=-1, name='normalize_y_true')
        y_pred = tf.reshape(y_pred, (-1, self._layers[-1]), name='reshape_y_pred')
        y_pred = tf.math.l2_normalize(y_pred, axis=-1, name='normalize_y_pred')
        dots = tf.reduce_sum(y_true * y_pred, axis=-1, name='dot_product')
        return tf.math.maximum(0., tf.math.subtract(1., dots, name='subtract_1'))
    
    def _negative_loss(self, y_true, y_pred):
        """Asymptotic graph near x=0 for negative sampling (https://www.desmos.com/calculator/uh9yklg2ij)."""
        y_true = tf.reshape(y_true, (-1, self._layers[-1]), name='reshape_y_true')
        y_true = tf.math.l2_normalize(y_true, axis=-1, name='normalize_y_true')
        y_pred = tf.reshape(y_pred, (-1, self._layers[-1]), name='reshape_y_pred')
        y_pred = tf.math.l2_normalize(y_pred, axis=-1, name='normalize_y_pred')
        dots = tf.reduce_sum(y_true * y_pred, axis=-1, name='dot_product')
        diff = tf.math.maximum(0., tf.math.subtract(1., dots, name='subtract_1'))
        return tf.math.divide(
                1.,
                tf.math.maximum(50 * diff, 1e-5)
        )
