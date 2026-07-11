from fastembed import TextEmbedding
import numpy as np

class EmbeddingModel:

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model = TextEmbedding(model_name=model_name)

    def encode(self,texts,convert_to_numpy: bool = True,
               normalize_embeddings: bool = True,
               batch_size: int = 8, 
               show_progress_bar: bool = False
               )-> np.ndarray:
        
        if isinstance(texts,str):
            texts = [texts]

        embeddings = list(self._model.embed(texts,batch_size=batch_size))
        arr = np.asarray(embeddings,dtype=np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(arr,axis=1,keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr/norms

        return arr
        