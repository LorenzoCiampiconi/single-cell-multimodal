from scmm.models.multioutput import MultiOutputRegressorMixin
from scmm.problems.cite.base import CiteModelABC
from scmm.models.lgbm import LGBMMixin
from scmm.models.embedding.concrete import MultiLevelEmbedderInputMixin, TruncateSVDEmbedderInputMixin


class LGBMwSVDCite(MultiOutputRegressorMixin, LGBMMixin, TruncateSVDEmbedderInputMixin, CiteModelABC):
    ...


class LGBMwMultilevelEmbedderCite(MultiOutputRegressorMixin, LGBMMixin, MultiLevelEmbedderInputMixin, CiteModelABC):
    ...
