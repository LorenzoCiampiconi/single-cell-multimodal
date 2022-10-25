from typing import Any, Dict
from scmm.models.base import MultiOutputRegressorMixin
from scmm.models.cite.base import CiteModelABC
from scmm.models.cite.lgbm import LGBMMixin
from scmm.models.cite.rf import RandomForestRegressorMixin
from scmm.models.embedding.concrete import MultiLevelEmbedderMixin, TruncateSVDEmbedderMixin, NOPEmbedderMixin


class LGBMwSVDCite(LGBMMixin, TruncateSVDEmbedderMixin, MultiOutputRegressorMixin, CiteModelABC):
    ...


class LGBMwMultilevelEmbedderCite(LGBMMixin, MultiLevelEmbedderMixin, MultiOutputRegressorMixin, CiteModelABC):
    ...

class RFCite(RandomForestRegressorMixin, NOPEmbedderMixin, MultiOutputRegressorMixin, CiteModelABC):
    ...