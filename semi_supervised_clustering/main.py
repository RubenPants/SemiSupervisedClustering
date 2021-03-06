"""Embedding model that embeds its inputs and assigns it to the right cluster, if cluster exists."""
from collections import Counter
from math import log
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector
from tqdm import tqdm

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
        self._thr = cluster_thr
        
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
    
    def __str__(self):
        """Representation of the model."""
        return f"EmbeddingModel"
    
    def __repr__(self):
        """Representation of the model."""
        return f"EmbeddingModel"
    
    def __call__(
            self,
            sentences: List[str],
            use_labeled: bool = True,
    ) -> List[Optional[str]]:
        """
        Define the best-suiting clusters for the provided sentences.
        
        :param sentences: The sentences to cluster
        :param use_labeled: If sentence comes literally from previously labeled sample then look up it's cluster ID
        :return: Cluster ID corresponding each of the sentences, None if no matching cluster
        """
        if not sentences: return []  # To prevent broken predictions
        predictions = self.clusterer(self.embed(sentences))
        if use_labeled:
            labeled = self.clusterer.get_all_labels()
            sentences = [self.clean_f(s) for s in sentences]
            return [labeled[s] if s in labeled else p for s, p in zip(sentences, predictions)]
        else:
            return predictions
    
    def embed(self, sentences: List[str]) -> np.ndarray:
        """Embed the list of sentences."""
        if not sentences: return np.zeros((0, self.embedder.dim), dtype=np.float32)  # To prevent breaks
        return self.embedder(self.encoder([self.clean_f(s) for s in sentences]))
    
    def get_cluster_prob(
            self,
            sentences: List[str],
            use_labeled: bool = True,
            use_labeled_none: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Cluster without a probability threshold and return closest cluster ID together with cosine similarity.
        
        :param sentences: Sentences for which the cluster probability gets computed
        :param use_labeled: Use pre-labeled data, and return with 100% match
        :param use_labeled_none: Included labeled-None data as well
        """
        if not sentences: return []  # To prevent broken predictions
        cluster_probs = self.clusterer.cluster_prob(self.embed(sentences))
        
        # Overwrite prediction if in labeled set
        if use_labeled:
            all_labels = self.clusterer.get_all_labels()
            if not use_labeled_none:
                all_labels = {k: v for k, v in all_labels.items() if v}
            for i, s in enumerate(sentences):
                if s in all_labels:
                    cluster_probs[i] = (all_labels[s], 1)
        return cluster_probs
    
    def get_all_cluster_prob(self, sentences: List[str], use_softmax: bool = False) -> Tuple[List[str], np.ndarray]:
        """Get softmax probabilities for all clusters."""
        if not sentences: return [], np.zeros((0, self.embedder.dim), dtype=np.float32)  # To prevent broken predictions
        return self.clusterer.all_clusters_prob(self.embed(sentences), use_softmax=use_softmax)
    
    def get_thr(self) -> float:
        """Get the clustering threshold."""
        return self._thr
    
    def initialise_models(
            self,
            data: List[str],
            reset_encoder: bool = False,
            reset_embedder: bool = False,
            reset_clusterer: bool = False,
            n_min_clusters: int = 5,
            show_overview: bool = False,
    ) -> None:
        """
        Initialise the models and annotate some clusters in order to start the training process.
        
        :param data: Data to initialise the models with
        :param reset_encoder: Reset previously progress made on the encoder, cannot be undone
        :param reset_embedder: Reset previously progress made on the embedder, cannot be undone
        :param reset_clusterer: Reset previously progress made on the clusters, cannot be undone
        :param n_min_clusters: Minimum of clusters to initialise
        :param show_overview: Show an overview for each of the model's components after initialisation
        """
        # Initialise the encoder
        if not self.encoder.is_created() or reset_encoder:
            self.encoder.create_encoder(data=data, show_overview=show_overview)
        
        # Reset the embedder, if requested
        if reset_embedder:
            self.embedder.create_model(show_overview=show_overview)
        
        # Reset the clusterer, if requested
        if reset_clusterer:
            self.clusterer.reset()
        if show_overview:
            self.clusterer.show_overview()
        
        if self.clusterer.get_cluster_count() < n_min_clusters:
            data_clean, data_count = self._transform_data(data=data)
            
            # Update embeddings using random negative sampling
            x, y = zip(*self.clusterer.sample_unsupervised(
                    n=1024 * 8,
                    items=data_clean,
                    embeddings=self.embed(data_clean),
            ))
            x = self.encoder.encode_batch(x, sample=True)
            y = np.vstack(y)
            switch = np.zeros((len(y), 1), dtype=np.float32)
            self.embedder.train(
                    x=x,
                    y=y,
                    switch=switch,
                    batch_size=1024,
            )
            
            # Create initial clusters
            updated = True
            embeddings = self.embed(data_clean)
            weights = [log(c) for c in data_count]
            while self.clusterer.get_cluster_count() < n_min_clusters or updated:
                score, proposal = self.clusterer.discover_new_cluster(
                        n=1,
                        items=data_clean,
                        embeddings=embeddings,
                        weights=weights,
                )[0]
                n_before = self.clusterer.get_cluster_count()
                self.clusterer.validate_cli(
                        item=proposal,
                        sim=score,
                )
                
                # Ensure to keep annotating as long as new clusters continue to get discovered
                updated = n_before != self.clusterer.get_cluster_count()
            print(f"\nFinished annotating new data samples")
    
    def train(
            self,
            data: List[str],
            batch_size: int = 1024,
            n_neg: int = 32 * 1024,
            n_pos: int = 32 * 1024,
            max_replaces: int = 10,
            epochs: int = 5,
            warmup: int = 8,
            iterations: int = 8,
            val_ratio: float = .1,
            n_val_cluster: int = 10,
            n_val_discover: int = 2,
            n_val_uncertain: int = 2,
            show_overview: bool = False,
            cli: bool = False,
            debug: bool = False,
    ) -> Any:
        """
        Train the embedding model using the supervised clusters.
        
        This training step re-shapes the global embedding-space by manipulating the embeddings associated with the
        supervised (known) cluster-data.
        
        :param data: Data on which is trained, this data is extended with the known cluster-data
        :param batch_size: Batch-size used during training
        :param n_neg: Number of negative samples sampled each iteration at most
        :param n_pos: Number of positive samples sampled each iteration at most
        :param max_replaces: Maximum number of replaces used during sampling (safety measure)
        :param epochs: Number of training/validation epochs (last epoch is not validated)
        :param warmup: Number of iterations to warmup the embeddings (run before validation-iterations)
        :param iterations: Number of iterations between validations
        :param val_ratio: Ratio used for validation during training
        :param n_val_cluster: Number of samples validated that are close to a cluster-boundary
        :param n_val_discover: Number of potential new clusters discovered during validation
        :param n_val_uncertain: Number of uncertain samples (those that may belong to more than one cluster) validated
        :param show_overview: Show/print overview of the training process each iteration
        :param cli: Validate using the CLI, if False the training will continue without validation
        :param debug: Print debugging information during training
        :return: Training history
        """
        if not self.clusterer.get_cluster_count():
            raise Exception("Initialise the models first, as well as some initial labels")
        
        # Initialise training
        data_clean, data_count = self._transform_data(data=data)
        history = None
        
        # Warmup the embeddings
        if warmup:
            pbar = tqdm(total=warmup, desc="Warmup, loss ???")
            try:
                for i in range(warmup):
                    hist = self._train_push_pull(
                            n_neg=n_neg,
                            n_pos=n_pos,
                            val_ratio=val_ratio,
                            batch_size=batch_size,
                            max_replaces=max_replaces,
                            debug=debug,
                    )
                    if not history:
                        history = hist
                    else:
                        for k in history.keys():
                            history[k] += hist[k]
                    pbar.set_description(f"Warmup, loss {round(history['loss'][-1], 5)}")
                    pbar.update()
            finally:
                pbar.close()
        
        # Train the model
        for epoch in range(1, epochs + 1):
            print(f"==> Running epoch {epoch} <==")
            
            # Transform embedding space
            pbar = tqdm(total=iterations, desc="Loss ???")
            try:
                for i in range(iterations):
                    hist = self._train_push_pull(
                            n_neg=n_neg,
                            n_pos=n_pos,
                            val_ratio=val_ratio,
                            batch_size=batch_size,
                            max_replaces=max_replaces,
                            debug=debug,
                    )
                    if not history:
                        history = hist
                    else:
                        for k in history.keys():
                            history[k] += hist[k]
                    pbar.set_description(f"Loss {round(history['loss'][-1], 5)}")
                    pbar.update()
            finally:
                pbar.close()
            
            # Set cluster-centroids using the embedder's current state
            self.clusterer.set_centroids(
                    embedding_f=self.embed,
            )
            
            # Calculate ratio of samples in cluster
            if show_overview or debug:
                predicted = self(data_clean, use_labeled=False)  # No cheating
                counter = Counter(predicted)
                print(f"\nTraining-clustering overview:")
                print(f" - Unclustered: {get_percentage(counter[None], sum(counter.values()))}")
                print(f" - Largest cluster: {max(v for k, v in counter.items() if k)}")
                print(f" - Average cluster: {sum(v for k, v in counter.items() if k) / len(counter)}")
                print(f" - Smallest cluster: {min(v for k, v in counter.items() if k)}")
                
                # Show validation overview (if validation set exists in cluster-data)
                self.validate(print_result=True)
            
            # Validate newly clustered samples (not on last epoch)
            if epoch != epochs and cli:
                self.clusterer.discover_unlabeled(
                        n_validate_cluster=n_val_cluster,
                        n_discover=n_val_discover,
                        n_uncertain=n_val_uncertain,
                        items=data_clean,
                        embeddings=self.embed(data_clean),
                        weights=data_count,  # Weight validation using item-counts
                        cli=cli,
                )
        
        # Store the trained model
        self.store()
        return history
    
    def validate(
            self,
            val_data: Optional[Tuple[str, str]] = None,
            print_result: bool = False,
    ) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """
        Validate the model.
        
        If no validation data is provided, the previously stored validation data (under clusterer) is used.
        
        :param val_data: Validation data (input, target_cluster)
        :param print_result: Print out the validation results
        :return: List of (input, true-label, predicted-label)
        """
        val_data = val_data if val_data else self.clusterer.get_validation_data()
        if not val_data:
            return []
        x, y = zip(*val_data)
        predicted_clusters = self(x, use_labeled=False)  # No cheating
        
        # Print the result if requested
        if print_result:
            print(f"\nValidation result:")
            n_correct = sum([true == pred for true, pred in zip(y, predicted_clusters)])
            print(f" - Global  Accuracy: {get_percentage(n_correct, len(x))}")
            n_cl = sum([true is not None for true in y])
            n_correct_cl = sum([pred is not None and true == pred for true, pred in zip(y, predicted_clusters)])
            print(f" - Accuracy cluster: {get_percentage(n_correct_cl, n_cl)}")
            n_correct_none = sum([pred is None and true == pred for true, pred in zip(y, predicted_clusters)])
            print(f" - Accuracy None   : {get_percentage(n_correct_none, len(x) - n_cl)}")
            n_unclustered = sum([pred is None for pred in predicted_clusters])
            print(f" - Number of predicted cluster: {get_percentage(len(x) - n_unclustered, len(x))}")
            print(f" - Number of predicted None   : {get_percentage(n_unclustered, len(x))}")
        return list(zip(x, y, predicted_clusters))
    
    def initialise_embeddings(
            self,
            data: List[str],
            embeddings: np.ndarray,
            iterations: int = 8,
            batch_size: int = 1024,
            n_replaces: int = 5,
    ) -> Any:
        """
        Initialise the embedding-model using pre-existing sentence embeddings.
        
        :param data: Sentence data used to train on
        :param embeddings: Pre-existing sentence embedding corresponding the sentence data
        :param iterations: Number of iterations between validations
        :param batch_size: Batch-size used during training
        :param n_replaces: Number of times the same data-sample is replaced during sampling
        :return: Training history
        """
        assert len(data) == len(embeddings)
        
        # Initialise fitting of provided embeddings
        data = [self.clean_f(d) for d in data] * n_replaces
        y = np.vstack([embeddings, ] * n_replaces)
        switch = np.ones((len(y), 1), dtype=np.float32)
        history = None
        
        # Fit pre-trained embeddings on provided data
        pbar = tqdm(total=iterations, desc="Loss ???")
        try:
            for i in range(iterations):
                x = self.encoder.encode_batch(data, sample=True)  # Re-sample every iteration
                hist = self.embedder.train(
                        x=x,
                        y=y,
                        switch=switch,
                        batch_size=batch_size,
                )
                if not history:
                    history = hist
                else:
                    for k in history.keys():
                        history[k] += hist[k]
                pbar.set_description(f"Loss {round(history['loss'][-1], 5)}")
                pbar.update()
        finally:
            pbar.close()
        return history
    
    def visualise_tensorboard(
            self,
            data: List[str],
            path_projector: Path,
            incl_none: bool = True,
            use_labeled: bool = True,
    ) -> None:
        """
        Visualise the resulting embeddings and clusters using TensorBoard.
        
        :param data: Data to embed and visualise
        :param path_projector: Path where projector-assets are stored
        :param incl_none: Include the samples that do not belong to a cluster
        :param use_labeled: If sentence comes literally from previously labeled sample then look up it's cluster ID
        """
        write_path = path_projector / f'{self.name}'
        write_path.mkdir(exist_ok=True, parents=True)
        
        # Get data embeddings and predict all the clusters
        embeddings = self.embed(data)
        representatives = self(data, use_labeled=use_labeled)
        
        # Filter out None-representatives if requested
        if not incl_none:
            data_f = []
            embeddings_f = []
            representatives_f = []
            for d, e, r in zip(data, embeddings, representatives):
                if r:  # Only add if representative not None
                    data_f.append(d)
                    embeddings_f.append(e)
                    representatives_f.append(r)
            data = data_f
            embeddings = embeddings_f
            representatives = representatives_f
        
        # Save Labels separately on a line-by-line manner.
        with open(write_path / f'metadata.tsv', "w") as f:
            joint = [f'{name}\t{rep}' for name, rep in zip(data, representatives)]
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
    
    def store(self):
        """Store all the sub-components of the model."""
        self.embedder.store()
        self.clusterer.store()
    
    def load(self):
        """Load in all the sub-components of the model."""
        self.embedder.load()
        self.encoder.load()
        self.clusterer.load()
    
    def _transform_data(
            self,
            data: List[str],
    ) -> Tuple[List[str], List[int]]:
        """Transform the data by cleaning, counting and adding missing cluster data."""
        # Clean and count provided data
        temp = []
        for d in data:
            c = self.clean_f(d)
            if c: temp.append(c)
        data_clean, data_count = zip(*sorted(Counter(temp).items(), key=lambda x: x[1]))
        data_clean = list(data_clean)
        data_count = list(data_count)
        
        # Append missing cluster-data with average frequency
        avg_freq = max(1, round(sum(data_count) / len(data_count)))
        for k, _ in self.clusterer.get_training_data() + self.clusterer.get_validation_data():
            if k not in data_clean:
                data_clean.append(k)
                data_count.append(avg_freq)
        return data_clean, data_count
    
    def _train_push_pull(
            self,
            n_neg: int,
            n_pos: int,
            val_ratio: float = .1,
            batch_size: int = 1024,
            max_replaces: int = 10,
            debug: bool = False,
    ) -> Any:
        """Perform a single push-pull training."""
        # Sample negative
        x_neg, y_neg = zip(*self.clusterer.sample_negative(
                n=n_neg,
                embedding_f=self.embed,
                max_replaces=max_replaces,
                debug=debug,
        ))
        
        # Sample positive
        x_pos, y_pos = zip(*self.clusterer.sample_positive(
                n=n_pos,
                embedding_f=self.embed,
                max_replaces=max_replaces,
                debug=debug,
        ))
        
        # Encode and format training inputs
        x = self.encoder.encode_batch(x_neg + x_pos, sample=True)
        y = np.vstack(y_neg + y_pos)
        switch = np.append(
                np.zeros((len(y_neg), 1), dtype=np.float32),
                np.ones((len(y_pos), 1), dtype=np.float32),
                axis=0,
        )
        
        # Train and return the history
        return self.embedder.train(
                x=x,
                y=y,
                switch=switch,
                batch_size=batch_size,
                val_ratio=val_ratio,
                debug=debug,
        )


def get_percentage(a, b) -> str:
    """Print percentage ratio of a/b."""
    return f"{round(100 * a / b, 2)}% ({a}/{b})"
