"""Cluster the found data."""
import json
from collections import Counter
from pathlib import Path
from random import choice, shuffle
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Clusterer:
    """Naive semi-supervised clustering implementation, that distinguishes known from estimated clusters."""
    
    def __init__(
            self,
            name: str,
            path_data: Path,
            path_model: Path,
            clean_f: Callable[..., str] = lambda x: x,
            cluster_thr: float = .9,
    ) -> None:
        """
        Naive semi-supervised clustering technique that clusters every item to exactly one cluster.

        This clustering implementation differentiates between known clusters and estimated clusters, as determined by
        the cosine similarity of the embedded items.

        :param name: Name of previously annotated clusters (cache)
        :param path_data: Path where training and validation data is stored
        :param path_model: Path where centroids are stored, used for both training and inference
        :param clean_f: Function used to clean the data before encoding it, only used by model call
        :param cluster_thr: Clustering threshold (cosine similarity)
        """
        self.clean_f = clean_f
        self._name = name
        self._sim_thr = cluster_thr
        self._path_data = path_data
        self._path_model = path_model
        self._clusters: Dict[str, Optional[str]] = {}  # Known clusters and labels
        self._clusters_val: Dict[str, Optional[str]] = {}  # Validation clusters and labels
        self._centroids: Dict[str, np.ndarray] = {}  # Cache for the cluster centroids
        
        # Load previously known clusters, if exists
        self.load()
    
    def __str__(self):
        """Cluster representation."""
        return f"cluster-{self._name}"
    
    def __repr__(self):
        """Cluster representation."""
        return str(self)
    
    def __call__(
            self,
            embeddings: np.ndarray,
    ) -> List[Optional[str]]:
        """Get the best-suiting clusters for the given embeddings, or None if no such cluster exists."""
        assert self._centroids  # Centroids must be set in advance
        
        # Setup known database
        cluster_ids, cluster_embs = zip(*self._centroids.items())
        cluster_embs = np.vstack(cluster_embs)
        
        # Calculate similarity with cluster centroids
        similarity = cosine_similarity(embeddings, cluster_embs)
        
        # Fetch the best-matching clusters
        results = []
        for i, idx in enumerate(similarity.argmax(axis=1)):
            results.append(cluster_ids[idx] if similarity[i, idx] >= self._sim_thr else None)
        return results
    
    def add_clusters(
            self,
            known_clusters: Dict[str, List[str]],
    ) -> None:
        """Add known clusters to the data, cluster-IDs that are None are considered cluster-less noise."""
        # Validate and add inputs
        for c_id, cluster in known_clusters.items():
            # Check if new cluster's items don't occur in other clusters
            possible_ids: Set[str] = set()
            for c in cluster:
                if c in self._clusters:
                    possible_ids.add(self._clusters[c])
            if possible_ids:
                # print(possible_ids)
                # print(c_id, cluster)
                assert len(possible_ids) <= 1
                assert list(possible_ids)[0] == c_id
            
            # Always add cluster ID as key, if cluster ID is not None
            if c_id:
                self._clusters[self.clean_f(c_id)] = c_id
            
            # Add to known clusters (extend to existing if possible, create new otherwise)
            for c in cluster:
                self._clusters[self.clean_f(c)] = c_id
        self.store()
    
    def add_to_cluster(
            self,
            item: str,
            c_id: Optional[str],
    ) -> None:
        """Add the given item to the requested cluster."""
        if item in self._clusters.keys():  # Check if conflicting add
            assert self._clusters[item] == c_id
        assert c_id is None or c_id in self._clusters.values()  # Cluster already exists
        self._clusters[item] = c_id
        self.store()
    
    def add_validation(
            self,
            val_clusters: Dict[str, List[str]],
    ) -> None:
        """Add data to the validation-set."""
        # Validate and add inputs
        for c_id, cluster in val_clusters.items():
            # Cluster-ID should also be present in training data (or be garbage-indicator)
            assert c_id is None or c_id in self._clusters.values()
            
            # Check if all cluster-values are not in training data and add to validation cluster
            for c in cluster:
                clean = self.clean_f(c)
                if clean in self._clusters.keys():
                    raise Exception(f"Sample '{c}' already in training set")
                self._clusters_val[clean] = c_id
        self.store()
    
    def get_training_data(self) -> List[Tuple[str, Optional[str]]]:
        """Return the validation data as (input_value, target_cluster_id)."""
        return [(k, v) for k, v in sorted(self._clusters.items(), key=lambda x: (x[1] if x[1] else '', x[0]))]
    
    def get_validation_data(self) -> List[Tuple[str, Optional[str]]]:
        """Return the validation data as (input_value, target_cluster_id)."""
        return [(k, v) for k, v in sorted(self._clusters_val.items(), key=lambda x: (x[1] if x[1] else '', x[0]))]
    
    def reset(self):
        """Reset clusterer by removing previously labeled data, cannot be undone."""
        self._clusters = {}
        self._clusters_val = {}
        self._centroids = {}
        self.store()
    
    def get_all_cluster_ids(self) -> Set[str]:
        """Get all the clusters."""
        return {v for v in self._clusters.values() if v}
    
    def get_all_clusters(self) -> Dict[str, List[str]]:
        """Get all the clusters (ID together with all their values)."""
        result = {}
        for c_id in set(self._clusters.values()):
            result[c_id] = self.get_cluster_by_id(c_id)
        return result
    
    def get_cluster_id(
            self,
            item: str,
    ) -> str:
        """Get the cluster-ID of the given item."""
        assert item in self._clusters.keys()
        return self._clusters[item]
    
    def get_cluster_by_id(
            self,
            c_id: str,
    ) -> List[str]:
        """Get all items from the cluster as specified by its ID."""
        return [k for k, v in self._clusters.items() if v == c_id]
    
    def get_cluster_count(self) -> int:
        """Count the number of clusters."""
        return len(self.get_all_cluster_ids())
    
    def get_centroids(
            self,
            items: List[str],
            embeddings: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Create cluster-centroids and update centroid cache."""
        centroids = {}
        relevant_items = [(idx, item) for idx, item in enumerate(items) if item in self._clusters.keys()]
        for c_id in self.get_all_cluster_ids():
            indices = [i for i, x in relevant_items if self._clusters[x] == c_id]
            assert indices != []
            centroids[c_id] = np.take(embeddings, indices, axis=0).mean(0)
        return centroids
    
    def set_centroids(
            self,
            items: List[str],
            embeddings: np.ndarray,
    ) -> None:
        """Set and store the final list of centroids, suited to the provided data."""
        self._centroids = self.get_centroids(
                items=items,
                embeddings=embeddings,
        )
    
    def sample_unsupervised(
            self,
            n: int,
            items: List[str],
            embeddings: np.ndarray,
            max_replaces: int = 10,
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Sample unsupervised any two different items at random, the second item is used as a repulsion-vector.

        :param n: Number of samples (upper limit, see n_replaces)
        :param items: Input-items to select from (assumed to be cleaned)
        :param embeddings: Embeddings used as repulsion-vectors
        :param max_replaces: Maximum number of times an item can be replaced during sampling
        :return: List of samples together with a random repulsion vector
        """
        self._centroids = {}  # Reset centroids when model gets trained
        
        # Create sampling list
        all_items = list(set(items)) * max_replaces
        shuffle(all_items)
        
        def get_other_item(item: str) -> np.ndarray:
            """Get the embedding of another item."""
            i = choice(range(len(items)))
            while items[i] == item:
                i = choice(range(len(items)))
            return embeddings[i]
        
        # Sample with (limited) replacement, unweighted sampling
        result = []
        for sample in np.random.choice(all_items, size=min(n, len(all_items)), replace=False):
            result.append((sample, get_other_item(sample)))
        return result
    
    def sample_positive(
            self,
            n: int,
            items: List[str],
            embeddings: np.ndarray,
            max_replaces: int = 10,
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Sample positive items from the clusters with their centroid as target-vector.

        :param n: Number of samples (upper limit, see max_replaces)
        :param items: Input-items to select from (assumed to be cleaned)
        :param embeddings: Embeddings used to determine the centroids
        :param max_replaces: Maximum number of times an item can be replaced during sampling
        :return: List of samples together with their corresponding cluster-centroid-embeddings
        """
        assert self._clusters
        self._centroids = {}  # Reset centroids when model gets trained
        
        # Update the cluster-centroids using the provided embeddings
        centroids = self.get_centroids(items=items, embeddings=embeddings)
        
        # Get all clusters containing more than one item (otherwise embedding is centroid)
        count = Counter(self._clusters.values())
        known_items_plural = [k for k, v in self._clusters.items() if v and count[v] > 1]
        known_items_plural *= max_replaces
        
        # Sample with (limited) replacement, unweighted sampling
        result = []
        for sample in np.random.choice(known_items_plural, size=min(n, len(known_items_plural)), replace=False):
            result.append((sample, centroids[self._clusters[sample]]))
        return result
    
    def sample_negative(
            self,
            n: int,
            items: List[str],
            embeddings: np.ndarray,
            max_replaces: int = 10,
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Sample negative items from the clusters together with a repulsion-vector.

        Every sampled couple consists of a labeled item (x) and a cluster-centroid (y) from a cluster different from the
        cluster to which x belongs. Note that the None-cluster doesn't have a centroid.

        :param n: Number of samples
        :param items: Input-items to select from (assumed to be cleaned)
        :param embeddings: Embeddings used to determine the centroids
        :param max_replaces: Maximum number of times an item can be replaced during sampling
        :return: List of samples together with their corresponding (repulsion) vector
        """
        assert self._clusters
        self._centroids = {}  # Reset centroids when model gets trained
        
        # Update the cluster-centroids using the provided embeddings
        centroids = self.get_centroids(items=items, embeddings=embeddings)
        
        # Enlist all sampling options
        known_items = list(set(self._clusters.keys()) & set(items))
        known_items *= max_replaces
        
        def get_repulsion_vector(item: str) -> np.ndarray:
            """Get a repulsion vector for the given item."""
            return centroids[choice([c_id for c_id in self.get_all_cluster_ids() if c_id != self._clusters[item]])]
        
        # Sample with (limited) replacement, unweighted sampling
        result = []
        for sample in np.random.choice(known_items, size=min(n, len(known_items)), replace=False):
            result.append((sample, get_repulsion_vector(sample)))
        return result
    
    def discover_unlabeled(
            self,
            n_validate_cluster: int,
            n_discover: int,
            n_uncertain: int,
            items: List[str],
            embeddings: np.ndarray,
            weights: Optional[List[float]] = None,
            cli: bool = False,
    ) -> Optional[List[Tuple[float, Tuple[str, Optional[str]]]]]:
        """
        Discover unlabeled items best to label next.
        
        Items that have a high chance of being proposed to get labeled are:
         - When close to a cluster-boundary (sim_thr)
         - When in a highly-dense unclustered area
         - When within the similarity threshold of more than one cluster
        
        :param n_validate_cluster: Number of samples generated close to cluster-boundary
        :param n_discover: Number of samples generated that may discover a new cluster
        :param n_uncertain: Number of samples generated that may occur in more than one cluster
        :param items: Items to consider
        :param embeddings: Item embeddings
        :param weights: Sampling weights corresponding each item, no weighting is applied if not provided
        :param cli: Use built-in CLI to validate the samples
        :return: If not validated via CLI; list of validations (similarity_score, (new_item, proposed_cluster))
        """
        self.set_centroids(items=items, embeddings=embeddings)
        validating_samples: List[Tuple[float, Tuple[str, Optional[str]]]] = []
        
        # Generate the items to validate
        validating_samples += self.discover_cluster_boundary(
                n=n_validate_cluster,
                items=items,
                embeddings=embeddings,
                weights=weights,
        )
        
        # Discover new clusters
        validating_samples += [(sim, (item, None)) for sim, item in self.discover_new_cluster(
                n=n_discover,
                items=items,
                embeddings=embeddings,
                weights=weights,
        )]
        
        # Discover items that can be placed in more than one cluster
        validating_samples += self.discover_uncertain(
                n=n_uncertain,
                items=items,
                embeddings=embeddings,
                weights=weights,
        )
        
        # Filter samples on duplicates and remove those that are present in _clusters_val
        added = set()
        temp = []
        for sim, (a, b) in validating_samples:
            if a not in added and a not in self._clusters_val.keys():
                added.add(a)
                temp.append((sim, (a, b)))
        validating_samples = temp
        
        # Validate the proposed items
        if cli:
            for sim, (a, b) in validating_samples:
                self.validate_cli(item=a, proposed_cluster=b, sim=sim)
            
            # Show progress and save the results
            self.show_overview()
            self.store()
        else:
            return validating_samples
    
    def discover_cluster_boundary(
            self,
            n: int,
            items: List[str],
            embeddings: np.ndarray,
            weights: Optional[List[float]] = None,
    ) -> List[Tuple[float, Tuple[str, str]]]:
        """
        Discover unlabeled items that are close to the clustering-boundary.

        To further simulate sampling, it is possible to provide additional weights to each item. The final sample
        weighting is then the product of the similarity weight (1 when on cluster-boundary, less else) and the provided
        weight. A suggested weighting would be one that correlates to the sample's frequency.

        :param n: Number of samples to generate
        :param items: Items to consider
        :param embeddings: Item embeddings
        :param weights: Sampling weights corresponding each item, no weighting is applied if not provided
        :return: A list of validation items, where each such item looks like (similarity, (unclustered,clustered))
        """
        # Generate weights if not provided, only consider un-clustered items
        weights = weights if weights else [1, ] * len(items)
        assert len(weights) == len(items)
        
        # Don't consider elements that are already validated
        known_items = set(self._clusters.keys())
        weights = [0 if items[i] in known_items else w for i, w in enumerate(weights)]
        
        # Calculate the similarities to all cluster-centroids
        cluster_ids, cluster_embs = zip(*self._centroids.items())
        cluster_embs = np.vstack(cluster_embs)
        
        # Calculate similarity with cluster centroids
        similarity = cosine_similarity(embeddings, cluster_embs)
        
        # For each item, get the certainty to its closest cluster-centroid
        item_similarities = []
        for i, idx in enumerate(similarity.argmax(axis=1)):
            item_similarities.append((similarity[i, idx], (items[i], cluster_ids[idx])))
        
        # Update the weights with the similarity-scores and sample unclustered validation items
        weights = [max(0., w * self._transform_sim(sim[0])) for w, sim in zip(weights, item_similarities)]
        chosen_indices = np.random.choice(
                range(len(items)),
                size=n,
                replace=False,
                p=np.asarray(weights, dtype='float32') / sum(weights),
        )
        return [item_similarities[idx] for idx in chosen_indices]
    
    def discover_new_cluster(
            self,
            n: int,
            items: List[str],
            embeddings: np.ndarray,
            weights: Optional[List[float]] = None,
            k_neighbours: int = 10,
    ) -> List[Tuple[float, str]]:
        """
        Discover a potential new cluster that doesn't yet belong to any existing cluster.

        :param n: Number of samples to generate
        :param items: Input-items to select from (assumed to be cleaned)
        :param embeddings: Embeddings used to determine the centroids
        :param weights: Weights applied on the sampling (to favor more frequent items)
        :param k_neighbours: Number of neighbours used to determine potential clusters
        :return: Tuple of (score, proposed_new_cluster)
        """
        # Get all cross-similarities
        similarity = cosine_similarity(embeddings)
        
        # Calculate scores for every row
        scores = []
        sorted_idx = similarity.argsort(axis=1)  # Get sorted indices (sort on corresponding values)
        for i, (item, weight) in enumerate(zip(items, weights)):
            # No point in calculating score if weight equals zero
            if not weight:
                scores.append(0)
                continue
            
            # Assign score of zero if labeled entity is in K nearest neighbours
            top_indices = sorted_idx[i, -k_neighbours:]
            if any(items[idx] in self._clusters.keys() for idx in top_indices):
                scores.append(0)
            
            # Use accumulated similarity of K nearest neighbours as score
            else:
                scores.append(weight * similarity[i, top_indices].sum())
        
        # Filter out the highest score item
        return list(sorted(zip(scores, items), key=lambda x: x[0], reverse=True))[:n]
    
    def discover_uncertain(
            self,
            n: int,
            items: List[str],
            embeddings: np.ndarray,
            weights: Optional[List[float]] = None,
    ) -> List[Tuple[float, Tuple[str, str]]]:
        """
        Discover items that can be placed in more than one cluster.

        :param n: Number of samples to generate
        :param items: Input-items to select from (assumed to be cleaned)
        :param embeddings: Embeddings used to determine the centroids
        :param weights: Weights applied on the sampling (to favor more frequent items)
        :return: A list of validation items, where each such item looks like (similarity, (unclustered,clustered))
        """
        # Generate weights if not provided, only consider un-clustered items
        weights = weights if weights else [1, ] * len(items)
        assert len(weights) == len(items)
        
        # Don't consider elements that are already validated
        known_items = set(self._clusters.keys())
        weights = [0 if items[i] in known_items else w for i, w in enumerate(weights)]
        
        # Calculate the similarities to all cluster-centroids
        cluster_ids, cluster_embs = zip(*self._centroids.items())
        cluster_embs = np.vstack(cluster_embs)
        
        # Calculate similarity with cluster centroids and sort
        similarity = cosine_similarity(embeddings, cluster_embs)
        sorted_idx = similarity.argsort(axis=1)
        
        # For each item, check if close to multiple clusters and get the certainty to its closest cluster-centroid
        item_similarities = []
        for i, (indices, w) in enumerate(zip(sorted_idx, weights)):
            second_best, best = indices[-2:]
            item_similarities.append((
                w * similarity[i, best] if similarity[i, second_best] >= self._sim_thr else 0,
                (items[i], cluster_ids[best])
            ))
        
        # Filter out those with a score greater than zero
        options = [(a, b) for a, b in item_similarities if a > 0]
        
        # Return all options if number of options less than desired sample-amount
        if len(options) <= n:
            return options
        
        # Sample options based on score
        weights = [a for a, _ in options]
        chosen_indices = np.random.choice(
                range(len(options)),
                size=n,
                replace=False,
                p=np.asarray(weights, dtype='float32') / sum(weights),
        )
        return [options[idx] for idx in chosen_indices]
    
    def show_overview(self) -> None:
        """Print the overview of the clusters' current state."""
        print(f"\n\nCluster overview:")
        all_clusters = self.get_all_clusters()
        print(f" - Total of {len(all_clusters)} clusters")
        if all_clusters:
            cluster_lengths = [len(v) for v in all_clusters.values()]
            print(f" - Average number of cluster-labels: {round(sum(cluster_lengths) / len(cluster_lengths), 2)}")
    
    def validate_cli(
            self,
            item: str,
            proposed_cluster: Optional[str] = None,
            sim: float = 0,
    ) -> None:
        """
        Validated the provided item via an CLI interface.

        There are five different options to choose from:
         - Add:a if the couple belongs to the same cluster, add it to this cluster
         - Help:h to show the complete cluster
         - Garbage:g if the unclustered item is considered garbage (i.e. shouldn't be clustered)
         - Ignore:i ignore if uncertain about which action to take best
         - Other:<other> to add to other/new cluster
        """
        print(f"\nAdd to cluster?{f' (sim: {round(sim, 3)})' if sim else ''}")
        print(f" -    Item: '{item}'")
        if proposed_cluster:
            print(f" - Cluster: '{proposed_cluster}'")
        
        try:
            options = sorted(self.get_all_cluster_ids())
            inp = input(f"Add:a, Help:h, Garbage:g, Ignore:i, Add to other/new cluster:<other>\n")
            if inp == 'a':
                self.add_to_cluster(item=item, c_id=proposed_cluster)
                print(f" --> Added to cluster '{proposed_cluster}'")
            elif inp == 'h':
                print(f" --> Listing all current clusters:")
                for o in options:
                    print(f"     - {o}")
                self.validate_cli(item=item, proposed_cluster=proposed_cluster, sim=sim)
            elif inp == 'g':
                self.add_to_cluster(item=item, c_id=None)
                print(f" --> Added as garbage")
            elif inp == 'i':
                print(f" --> Ignored")
            else:
                if inp in options:
                    self.add_to_cluster(item=item, c_id=inp)
                    print(f" --> Added to cluster '{inp}'")
                else:
                    print(f" --> You're creating a new cluster '{inp}'")
                    confirmation = input(" --> Continue? (Yes:y, No:<other>) : ")
                    if confirmation == 'y':
                        self.add_clusters({inp: [item]})
                        print(f" --> Creating new cluster '{inp}'")
                    else:
                        print(f" --> Trying again")
                        self.validate_cli(item=item, proposed_cluster=proposed_cluster, sim=sim)
        except AssertionError as e:
            print(e)
            print(f" --> Invalid action, try again")
            self.validate_cli(item=item, proposed_cluster=proposed_cluster, sim=sim)
    
    def store(self) -> None:
        """Store the current centroids, training and validation data."""
        # Store the centroids
        with open(self._path_model / f"{self}", 'w') as file:
            json.dump({k: v.tolist() for k, v in self._centroids.items()}, file, sort_keys=True)
        
        # Store the (validation) clusters
        with open(self._path_data / f"{self}-train", 'w') as file:
            json.dump(self._clusters, file, indent=2, sort_keys=True)
        with open(self._path_data / f"{self}-val", 'w') as file:
            json.dump(self._clusters_val, file, indent=2, sort_keys=True)
    
    def load(self) -> None:
        """Load in previously-created centroids, training and validation data."""
        # Load in centroids
        if (self._path_model / f"{self}").is_file():
            with open(self._path_model / str(self), 'r') as file:
                self._centroids = {k: np.asarray(v, dtype=np.float32) for k, v in json.load(file).items()}
        
        # Load in (validation) clusters
        if (self._path_data / f"{self}-train").is_file():
            with open(self._path_data / f"{self}-train", 'r') as file:
                self._clusters = json.load(file)
        if (self._path_data / f"{self}-val").is_file():
            with open(self._path_data / f"{self}-val", 'r') as file:
                self._clusters_val = json.load(file)
    
    def _transform_sim(self, x: float) -> float:
        """
        Transform the similarity-score x to be maximal when close to the threshold, and lower else.
        
        This follows the assumption that x is between 0 and 1, where 1 is the cluster's center.
        
        :param x: Similarity-score
        :return: Transformed score
        """
        x = max(0.0, x)
        max_diff = self._sim_thr if x < self._sim_thr else (1 - self._sim_thr)
        return max(0., 1. - abs(x - self._sim_thr) / max_diff)
