from scmm.models.output_embedder import ODRwSVDMixin
from scmm.models.embedding.concrete import TruncateSVDEmbedderInputMixin, MultiLevelEmbedderInputMixin
from scmm.models.lgbm import LGBMMixin
from scmm.models.multioutput import MultiOutputRegressorMixin
from scmm.models.ridge import RidgeMixin
from scmm.problems.multiome.base import MultiomeModelABC


class RidgeSVDMulti(RidgeMixin, TruncateSVDEmbedderInputMixin, MultiomeModelABC):
    ...


class RidgeSVDwAutoencoderMulti(RidgeMixin, MultiLevelEmbedderInputMixin, MultiomeModelABC):
    ...


class SVDinLGBMOutSVDMultiome(
    ODRwSVDMixin, MultiOutputRegressorMixin, LGBMMixin, TruncateSVDEmbedderInputMixin, MultiomeModelABC
):
    ...


class SVDwAutoencoderInLGBMOutSVDMultiome(
    ODRwSVDMixin, MultiOutputRegressorMixin, LGBMMixin, MultiLevelEmbedderInputMixin, MultiomeModelABC
):
    ...
