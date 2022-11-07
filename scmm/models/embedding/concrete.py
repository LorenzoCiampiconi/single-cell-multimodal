from typing import Type
from scmm.models.embedding.base_embedder import Embedder, EmbedderWrapperInputMixin
from scmm.models.embedding.multilevel import MultiLevelEmbedder
from scmm.models.embedding.svd import TruncatedSVDEmbedder


class TruncateSVDEmbedderInputMixin(EmbedderWrapperInputMixin):
    @property
    def embedder_class(self) -> Type[Embedder]:
        return TruncatedSVDEmbedder


class MultiLevelEmbedderInputMixin(EmbedderWrapperInputMixin):
    @property
    def embedder_class(self) -> Type[Embedder]:
        return MultiLevelEmbedder
