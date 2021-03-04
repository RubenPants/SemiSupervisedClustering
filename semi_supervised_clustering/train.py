"""Train the custom embedding model together with its corresponding encoder and clustering class."""
from collections import Counter
from math import sqrt
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from tensorboard.plugins import projector
from tqdm import tqdm

from semi_supervised_clustering.cluster import Clusterer
from semi_supervised_clustering.embed import Embedder
from semi_supervised_clustering.encode import Encoder


# TODO: Deprecated; remove!
class Trainer:
    """Train a requested collection (model, encoder, and cluster)."""
    
    def __init__(
            self,
            name: str,
            data: List[str],
            clusters: Dict[str, List[str]],
            clean_f: Callable[..., str] = lambda x: x,
            min_freq: int = 2,
            vocab_size: int = 300,
            encoder_type: str = 'bpe',
            model_layers: List[int] = (100, 100,),
            normalise: bool = True,
            cluster_thr: float = .9,
            garbage: Optional[Set[str]] = None,
            path_data: Path = Path(__file__).parent / '../data',
            path_model: Path = Path(__file__).parent / '../models',
            path_projector: Path = Path(__file__).parent / '../projector',
    ) -> None:
        """
        Initialise the training class.
        
        This class functions as a wrapper around the EmbeddingModel, Encoder, and Clustering classes to train them
        alongside each other.
        
        :param name: Name of the models to train
        :param data: The data to train on
        :param clusters: Predefined clusters that exist within the data
        :param vocab_size: Size of the input-vocabulary
        :param clean_f: Function used to clean the raw data
        :param min_freq: Minimal frequency of the cleaned data before being considered
        :param encoder_type: Type of SentencePiece encoder used
        :param model_layers: Number of layers present in the model, last layer is the embedding
        :param normalise: Normalise the output embeddings
        :param cluster_thr: Threshold before assigning a value to a specific cluster
        :param garbage: Optionally garbage added to generalise (prevent False Positives)
        :param path_data: Path to where data is stored
        :param path_model: Path to where models are stored
        :param path_projector: Path to where projecting embeddings
        """
        # General training parameters
        self.name = name
        self.path_data = path_data
        self.path_model = path_model
        self.path_projector = path_projector
        self.embeddings: Optional[np.ndarray] = None
        
        # Manipulate the incoming data
        data = [clean_f(d) for d in data]
        
        # Add cluster-data to data if not present
        avg_freq = max(round(len(data) / len(set(data))), 1)
        for key, values in clusters.items():
            for val in values:
                c = data.count(val)
                if c < avg_freq:
                    data += [clean_f(val), ] * (avg_freq - c)
        
        # Assure that garbage is cleaned
        self.garbage = {clean_f(g) for g in garbage} if garbage else set()
        
        # Count word-occurrences
        self.items, self.counts = zip(*[(item, count) for item, count in Counter(data).items() if count >= min_freq])
        print(f"Total of {len(data)} data items (frequency>={min_freq}):")
        print(f" -->  Unique items: {len(self.items)}")
        print(f" --> Max frequency: {max(self.counts)}")
        print(f" --> Avg frequency: {round(len(data) / len(self.items), 2)}")
        print(f" --> Med frequency: {sorted(self.counts)[len(self.counts) // 2]}")
        print(f" --> Min frequency: {min(self.counts)}")
        
        # Setup the encoder
        self.encoder = Encoder(
                name=name,
                clean_f=clean_f,
                vocab_size=vocab_size,
                model_type=encoder_type,
                path=self.path_model,
        )
        
        # Train encoder on data (note: this overwrites previous version of the model)
        self.encoder.create_encoder(
                data=[i * round(sqrt(c)) for i, c in zip(self.items, self.counts)],
        )
        self.encoder.analyse(
                words=self.items[:5]
        )
        
        # Load in the embedding-model
        self.model = Embedder(
                name=name,
                input_size=vocab_size,
                layers=model_layers,
                normalise=normalise,
                path=self.path_model,
        )
        
        # Setup the clustering-class
        self.cluster = Clusterer(
                name=name,
                cluster_thr=cluster_thr,
                path=self.path_data,
        )
        
        # Synchronise the clusters with the current data
        self.cluster.synchronise(data=data)
        
        # Add initial set of clusters and garbage
        self.cluster.add_known_clusters(known_clusters=clusters)
        self.cluster.add_garbage(garbage=self.garbage)
        self.cluster.print_overview()
    
    def __str__(self):
        """Trainer representation."""
        return f"Training({self.name})"
    
    def __repr__(self):
        """Trainer representation."""
        return str(self)
    
    def train(
            self,
            batch_size: int = 2048,
            n_neg: int = 32 * 2048,
            n_pos: int = 16 * 2048,
            epochs: int = 5,
            iterations: int = 8,
            n_validations: int = 20,
            show_overview: bool = True,
    ) -> None:
        """Core training algorithm."""
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
        
        # Plot the training overview if requested
        if show_overview:
            self.cluster.print_overview()
            loss_neg, loss_pos = zip(*loss_split)
            plt.figure(figsize=(10, 8))
            ax = plt.subplot(3, 1, 1)
            plt.plot(loss)
            plt.title("Average loss")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            ax = plt.subplot(3, 2, 3)
            plt.plot(loss_neg)
            plt.title("Negative-sample loss")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            ax = plt.subplot(3, 2, 4)
            plt.plot(loss_pos)
            plt.title("Positive-sample loss")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            ax = plt.subplot(3, 1, 3)
            plt.plot(cluster_count)
            plt.title("Number of clusters")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            plt.tight_layout()
            plt.show()
    
    def _train_push_pull(
            self,
            n_neg: int,
            n_pos: int,
            batch_size: int,
    ) -> Tuple[float, Tuple[float, float]]:
        """Perform a single push-pull training."""
        # Negative sampling
        x, y = zip(*self.cluster.sample_negative(
                n=n_neg,
                items=self.items,
                embeddings=self.embeddings,
        ))
        x = self.encoder.encode_batch(x, sample=True)
        y = np.vstack(y)
        loss_neg = self.model.train_negative(x=x, y=y, batch_size=batch_size)
        self._update_embeddings()
        
        # Positive sampling
        x, y = zip(*self.cluster.sample_positive(
                n=n_pos,
                items=self.items,
                embeddings=self.embeddings,
        ))
        x = self.encoder.encode_batch(x, sample=True)
        y = np.vstack(y)
        loss_pos = self.model.train_positive(x=x, y=y, batch_size=batch_size)
        self._update_embeddings()
        return (loss_neg + loss_pos) / 2, (loss_neg, loss_pos)  # Return the average loss
    
    def _update_embeddings(self) -> None:
        """Update the embeddings of the item-list, together with the approximated clusters."""
        self.embeddings = self.model(self.encoder(self.items))  # TODO: Right to do it here?
    
    def visualise_tensorboard(self) -> None:
        """Visualise the embedded words in TensorBoard."""
        write_path = self.path_projector / f'{self.name}'
        write_path.mkdir(exist_ok=True)
        
        # Predict all the clusters
        self.embeddings = self.model(self.encoder(self.items))  # Assure current embeddings are used
        representatives = self.cluster(self.items, similarity=cosine_similarity(self.embeddings))
        
        # Save Labels separately on a line-by-line manner.
        with open(write_path / f'metadata.tsv', "w") as f:
            joint = [f'{name}\t{rep}' for name, rep in zip(self.items, representatives)]
            f.write('Name\tCluster\n' + '\n'.join(joint) + '\n')
        
        # Save the weights we want to analyse as a variable. Note that the first value represents any unknown word,
        # which is not in the metadata, so we will remove that value.
        weights = tf.Variable(self.embeddings)
        
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
