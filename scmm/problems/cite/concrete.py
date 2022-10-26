from scmm.models.base import MultiOutputRegressorMixin
from scmm.problems.cite.base import CiteModelABC
from scmm.models.lgbm import LGBMMixin
from scmm.models.embedding.concrete import MultiLevelEmbedderMixin, TruncateSVDEmbedderMixin


class LGBMwSVDCite(LGBMMixin, TruncateSVDEmbedderMixin, MultiOutputRegressorMixin, CiteModelABC):
    ...


class LGBMwMultilevelEmbedderCite(LGBMMixin, MultiLevelEmbedderMixin, MultiOutputRegressorMixin, CiteModelABC):
    ...
