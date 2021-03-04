"""Embedding model that embeds its inputs and assigns it to the right cluster, if cluster exists."""
from collections import Counter
from math import log
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from tensorboard.plugins import projector

from semi_supervised_clustering.cluster import Clusterer
from semi_supervised_clustering.embed import Embedder
from semi_supervised_clustering.encode import Encoder


class EmbeddingModel:
    """Create an embedding model using semi-supervised clustering."""
    
    def __init__(
            self,
            name: str,
            path_model: Path,
            path_data: Path,
            clean_f: Callable[..., str] = lambda x: x,
            vocab_size: int = 300,
            encoder_type: str = 'bpe',
            model_layers: List[int] = (100, 100,),
            attention: bool = True,
            normalise: bool = True,
            cluster_thr: float = .9,
    ) -> None:
        """
        Initialise the main components of the embedding model.
        
        There are three main components out of which the EmbeddingModel is made of:
         - Encoder - encodes the raw string inputs
         - Embedder - embeds the encoded values into embeddings (vectors)
         - Clusterer - clusters the embeddings and assigns the best fitting cluster-ID to them
         
        :param name: Name of the embedding model
        :param path_model: Path indicating where the models are stored (directory)
        :param path_data: Path indicating where the data used during training/validation is stored (directory)
        :param clean_f: Cleaning function that transforms/cleans the raw text inputs
        :param vocab_size: Size of the encoder's vocabulary (which is also the embedder's input-size)
        :param encoder_type: Type of SentencePiece encoder used
        :param model_layers: Dense layers of the model, in chronological order (last layer is the embedding space)
        :param attention: Whether or not to perform attention on the model's layers (input as query)
        :param normalise: Whether or not to normalise the model's outputs
        :param cluster_thr: Similarity threshold (cosine) to exceed before getting assigned to a cluster
        """
        # General training parameters
        self.name = name
        self.clean_f = clean_f
        self.path_model = path_model
        self.path_model.mkdir(exist_ok=True, parents=True)
        self.path_data = path_data
        self.path_data.mkdir(exist_ok=True, parents=True)
        
        # Load in the encoder
        self.encoder = Encoder(
                name=name,
                clean_f=clean_f,
                vocab_size=vocab_size,
                model_type=encoder_type,
                path=self.path_model,
        )
        
        # Load in the embedding-model
        self.embedder = Embedder(
                name=name,
                input_size=vocab_size,
                layers=model_layers,
                attention=attention,
                normalise=normalise,
                path=self.path_model,
        )
        
        # Load in the clustering-class
        self.clusterer = Clusterer(
                name=name,
                clean_f=clean_f,
                cluster_thr=cluster_thr,
                path_model=self.path_model,
                path_data=self.path_data,
        )
    
    def __call__(self, sentences: List[str]) -> List[Optional[str]]:
        """Define the best-suiting clusters for the provided sentences."""
        # Embed the new sentences first
        embeddings = self.embedder(self.encoder([self.clean_f(s) for s in sentences]))
        return self.clusterer(embeddings)
    
    def __str__(self):
        """Representation of the model."""
        return f"EmbeddingModel"
    
    def __repr__(self):
        """Representation of the model."""
        return f"EmbeddingModel"
    
    def initialise_models(
            self,
            data: List[str],
            reset: bool = False,
            n_min_clusters: int = 3,
            show_overview: bool = True,
    ):
        """
        Initialise the models.
        
        :param data: Data to initialise the models with
        :param reset: Reset previously progress made, cannot be undone
        :param n_min_clusters: Minimum of clusters to initialise
        :param show_overview: Show an overview for each of the model's components after initialisation
        """
        # Clean the data in advance
        data = [self.clean_f(d) for d in data]
        
        # Initialise the encoder
        if not self.encoder.is_created() or reset:
            self.encoder.create_encoder(data=data, show_overview=show_overview)
        
        # Reset the embedder, if requested
        if reset:
            self.embedder.create_model(show_overview=show_overview)
        
        # Reset the clusterer, if requested
        if reset:
            self.clusterer.reset()
        self.clusterer.synchronise(data)
        if show_overview:
            self.clusterer.show_overview()
        
        if self.clusterer.get_cluster_count() < n_min_clusters:
            # Update embeddings using random negative sampling
            items, counts = zip(*Counter(data).items())
            x, y = zip(*self.clusterer.sample_unsupervised(
                    n=1024 * 8,
                    items=items,
                    embeddings=self.embedder(self.encoder(items)),
            ))
            x = self.encoder.encode_batch(x, sample=True)
            y = np.vstack(y)
            self.embedder.train_negative(x=x, y=y, batch_size=1024)
            
            # Create initial clusters
            while self.clusterer.get_cluster_count() < n_min_clusters:
                score, proposal = self.clusterer.discover_new_cluster(
                        n=1,
                        items=items,
                        embeddings=self.embedder(self.encoder(items)),
                        weights=[log(c) for c in counts],
                )[0]
                self.clusterer.validate_cli(
                        item=proposal,
                        sim=score,
                )
    
    def train(
            self,
            data: List[str],
            reset: bool = False,
            batch_size: int = 2048,
            n_neg: int = 32 * 2048,
            n_pos: int = 16 * 2048,
            epochs: int = 5,
            iterations: int = 8,
            n_validations: int = 20,
            show_overview: bool = True,
    ) -> None:
        """Core training algorithm."""  # TODO
        # Initialise the models if not yet done
        self.initialise_models(reset=reset)
        
        raise Exception
        
        # TODO: Proceed
        
        # TODO: Set cluster centroids at end of training
        
        # TODO: Initialise from train.py
        # Capture current state (embedding and cluster)
        self._update_embeddings()
        self.cluster.print_overview()
        loss, loss_split, cluster_count = [], [], [self.cluster.get_cluster_count(incl_approx=True), ]
        
        # Train the model
        for epoch in range(1, epochs + 1):
            print(f"==> Running epoch {epoch} <==")
            
            # Transform embedding space
            pbar = tqdm(total=iterations, desc="Loss ???")
            try:
                for i in range(iterations):
                    a, b = self._train_push_pull(
                            n_neg=n_neg,
                            n_pos=n_pos,
                            batch_size=batch_size,
                    )
                    loss.append(a)
                    loss_split.append(b)
                    cluster_count.append(self.cluster.get_cluster_count(incl_approx=True))
                    pbar.set_description(f"Loss {round(loss[-1], 5)}")
                    pbar.update()
            finally:
                pbar.close()
            
            # Validate newly clustered samples
            if epoch != epochs:  # Don't validate on the last epoch
                print(f"Validating:")
                self.cluster.validate(
                        n=n_validations,
                        items=self.items,
                        similarity=cosine_similarity(self.embeddings),
                        weights=self.counts,  # Weight validation using item-counts
                )
            
            # Calculate ratio of samples in cluster
            if show_overview:
                cluster_ids = self.cluster(
                        items=self.items,
                        similarity=cosine_similarity(self.embeddings),
                )
                n_clustered = 0
                for i, count in enumerate(self.counts):
                    if cluster_ids[i]:
                        n_clustered += count
                print(f"\nRatio of clustered items: {round(100 * n_clustered / sum(self.counts), 2)}%")
                n_non_garbage = sum(c for i, c in zip(self.items, self.counts) if i not in self.garbage)
                print(f"Ratio of clustered non-garbage items: {round(100 * n_clustered / n_non_garbage, 2)}%")
        
        # Store the trained model
        self.model.store_model()
    
    def initialise_embeddings(
            self,
            data: List[str],
            embeddings: np.ndarray,
            batch_size: int = 1024,
            n_replaces: int = 5,
    ) -> float:
        """
        Initialise the embedding-model using pre-existing sentence embeddings.
        
        :param data: Sentence data used to train on
        :param embeddings: Pre-existing sentence embedding corresponding the sentence data
        :param batch_size: Batch-size used during training
        :param n_replaces: Number of times the same data-sample is sampled during training
        :return: Final training loss
        """
        assert len(data) == len(embeddings)
        data *= n_replaces
        x = self.encoder.encode_batch(data, sample=True)
        y = np.vstack([embeddings, ] * n_replaces)
        loss = self.embedder.train_positive(x=x, y=y, batch_size=batch_size)
        return loss
    
    def _train_push_pull(
            self,
            data: List[str],
            n_neg: int,
            n_pos: int,
            batch_size: int = 1024,
            n_replaces: int = 5,
    ) -> Tuple[float, Tuple[float, float]]:
        """Perform a single push-pull training."""
        embeddings = None  # TODO: Use model to get embeddings for data
        # Negative sampling
        x, y = zip(*self.clusterer.sample_negative(
                n=n_neg,
                items=data,
                embeddings=embeddings,
                n_replaces=n_replaces,
        ))
        x = self.encoder.encode_batch(x, sample=True)
        y = np.vstack(y)
        loss_neg = self.embedder.train_negative(x=x, y=y, batch_size=batch_size)
        
        # Positive sampling
        embeddings = None  # TODO: Use model to get embeddings for data
        x, y = zip(*self.clusterer.sample_positive(
                n=n_pos,
                items=data,
                embeddings=embeddings,
                n_replaces=n_replaces,
        ))
        x = self.encoder.encode_batch(x, sample=True)
        y = np.vstack(y)
        loss_pos = self.embedder.train_positive(x=x, y=y, batch_size=batch_size)
        return (loss_neg + loss_pos) / 2, (loss_neg, loss_pos)  # Return the average loss
    
    def visualise_tensorboard(
            self,
            items: List[str],
            path_projector: Path,
    ) -> None:
        """TODO"""
        write_path = path_projector / f'{self.name}'
        write_path.mkdir(exist_ok=True)
        
        # Predict all the clusters
        embeddings = self.model(self.encoder(items))  # Assure current embeddings are used
        representatives = self.cluster(items, similarity=cosine_similarity(embeddings))
        
        # Save Labels separately on a line-by-line manner.
        with open(write_path / f'metadata.tsv', "w") as f:
            joint = [f'{name}\t{rep}' for name, rep in zip(items, representatives)]
            f.write('Name\tCluster\n' + '\n'.join(joint) + '\n')
        
        # Save the weights we want to analyse as a variable. Note that the first value represents any unknown word,
        # which is not in the metadata, so we will remove that value.
        weights = tf.Variable(embeddings)
        
        # Create a checkpoint from embedding, the filename and key are name of the tensor.
        checkpoint = tf.train.Checkpoint(embedding=weights)
        checkpoint.save(write_path / f"embedding.ckpt")
        
        # Set up config
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        
        # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = f'metadata.tsv'
        projector.visualize_embeddings(write_path, config)
        
        print("Run tensorboard in terminal:")
        print(f"tensorboard --logdir {write_path}")
        
        print("\nOr run tensorboard in notebook:")
        print(f"%load_ext tensorboard")
        print(f"%tensorboard --logdir {write_path}")
