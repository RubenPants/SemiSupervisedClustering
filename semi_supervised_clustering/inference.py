"""Load in a model, corresponding encoder and clustering classes to perform inference."""
from pathlib import Path
from typing import Callable, List, Optional

from sklearn.metrics.pairwise import cosine_similarity

from semi_supervised_clustering.cluster import Clusterer
from semi_supervised_clustering.embed import Embedder
from semi_supervised_clustering.encode import Encoder


# TODO: Deprecated; remove!
class CustomModel:  # TODO: Add in main
    def __init__(
            self,
            name: str,
            clean_f: Callable[..., str] = lambda x: x,
            vocab_size: int = 300,
            encoder_type: str = 'bpe',
            model_layers: List[int] = (100, 100,),
            normalise: bool = True,
            cluster_thr: float = .9,
            path_data: Path = Path(__file__).parent / '../data',
            path_model: Path = Path(__file__).parent / '../models',
    ) -> None:
        """TODO"""
        # General training parameters
        self.name = name
        self.clean_f = clean_f
        self.path_data = path_data
        self.path_model = path_model
        
        # Load in the encoder
        self.encoder = Encoder(
                name=name,
                clean_f=clean_f,
                vocab_size=vocab_size,
                model_type=encoder_type,
                path=self.path_model,
        )
        
        # Load in the embedding-model
        self.model = Embedder(
                name=name,
                input_size=vocab_size,
                layers=model_layers,
                normalise=normalise,
                path=self.path_model,
        )
        
        # Load in the clustering-class
        self.cluster = Clusterer(
                name=name,
                cluster_thr=cluster_thr,
                path=self.path_data,
        )
        
        # Embed all known clusters in advance
        clusters = self.cluster.get_all_clusters()
        self.known_items = [v for value in clusters.values() for v in value]
        self.known_embs = self.model(self.encoder(self.known_items))
    
    def __call__(
            self,
            sentences: List[str],
            sim_thr=.9,
    ) -> List[Optional[str]]:
        """TODO: Get milestones for list of sentences"""
        # Embed the new sentences first
        embeddings = self.model(self.encoder([self.clean_f(s) for s in sentences]))
        
        # Calculate similarity with known items
        similarity = cosine_similarity(embeddings, self.known_embs)
        
        # Fetch the best-matching clusters
        results = []
        for sim in similarity:
            best_idx = sim.argmax()
            results.append(
                    self.cluster.get_cluster_id(self.known_items[best_idx]) if sim[best_idx] >= sim_thr else None
            )
        return results
