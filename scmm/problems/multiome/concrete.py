from scmm.models.embedding.concrete import TruncateSVDEmbedderMixin
from scmm.models.ridge import RidgeMixin
from scmm.problems.multiome.base import MultiomeModelABC


class RidgeSVDMulti(RidgeMixin, TruncateSVDEmbedderMixin, MultiomeModelABC):
    ...