from typing import Type
from scmm.models.embedding.base import Embedder, EmbedderWrapperMixin
from scmm.models.embedding.multilevel import MultiLevelEmbedder
from scmm.models.embedding.svd import TruncatedSVDEmbedder


class TruncateSVDEmbedderMixin(EmbedderWrapperMixin):
    @property
    def embedder_class(self) -> Type[Embedder]:
        return TruncatedSVDEmbedder


class MultiLevelEmbedderMixin(EmbedderWrapperMixin):
    @property
    def embedder_class(self) -> Type[Embedder]:
        return MultiLevelEmbedder
