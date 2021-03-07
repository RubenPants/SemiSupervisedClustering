"""Create a mapping to another dimensionality such that the cosine-similarities hold as good as possible."""
from typing import List, Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class Mapper:
    """Mapping model to transform embedding dimensionality."""
    
    def __init__(
            self,
            inp_size: int,
            out_size: int,
    ) -> None:
        """Initialise the mapping-model's attributes."""
        self.inp_size = inp_size
        self.out_size = out_size
        self._model: Optional[tf.keras.Model] = None
    
    def __call__(
            self,
            embeddings: np.ndarray,
    ) -> np.ndarray:
        """Transform the embeddings to the requested size."""
        assert self._model is not None
        return self._model(embeddings)
    
    def train(
            self,
            embeddings: np.ndarray,
            iterations: int = 16,
            batch_size: int = 2048,
            sample_size: int = 64 * 2048,
    ) -> List[float]:
        """Create and train the mapping model."""
        # Create the model used to learn mapping
        inp1 = tf.keras.Input(
                shape=(self.inp_size,),
                name='Input-1',
        )
        inp2 = tf.keras.Input(
                shape=(self.inp_size,),
                name='Input-2',
        )
        dense_map = tf.keras.layers.Dense(
                self.out_size,
                activation='tanh',
                name='Mapper'
        )
        x1 = dense_map(inp1)
        x2 = dense_map(inp2)
        dotted = tf.keras.layers.Dot(
                axes=1,
                normalize=True,
                name="Dot",
        )([x1, x2])
        mapper_train = tf.keras.Model(
                inputs=[inp1, inp2],
                outputs=dotted,
                name="Mapper-Train",
        )
        mapper_train.compile(
                optimizer='adam',
                loss='mse',
        )
        
        # Train the mapper
        loss = []
        indices = list(range(len(embeddings)))
        similarity = cosine_similarity(embeddings)
        pbar = tqdm(total=iterations, desc="Loss ???")
        try:
            for _ in range(iterations):
                idx1 = np.random.choice(indices, size=sample_size)
                emb1 = np.vstack([embeddings[idx] for idx in idx1])
                idx2 = np.random.choice(indices, size=sample_size)
                emb2 = np.vstack([embeddings[idx] for idx in idx2])
                targets = np.asarray([similarity[f, t] for f, t in zip(idx1, idx2)])
                loss.append(mapper_train.fit(
                        x=[emb1, emb2],
                        y=targets,
                        batch_size=batch_size,
                        verbose=0,
                ).history['loss'][0])
                pbar.set_description(f"Loss {round(loss[-1], 5)}")
                pbar.update()
        finally:
            pbar.close()
        
        # Combine useful layers into final model
        self._model = tf.keras.Model(
                inputs=inp1,
                outputs=x1,
                name="Mapper"
        )
        self._model.summary()
        return loss
