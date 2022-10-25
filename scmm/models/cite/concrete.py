from typing import Any, Dict
from scmm.models.base import MultiOutputRegressorMixin
from scmm.models.cite.base import CiteModelABC
from scmm.models.cite.lgbm import LGBMMixin
from scmm.models.embedding.concrete import MultiLevelEmbedderMixin, TruncateSVDEmbedderMixin


class LGBMwSVDCite(LGBMMixin, TruncateSVDEmbedderMixin, MultiOutputRegressorMixin, CiteModelABC):
    ...


class LGBMwSVDAutoEncoderCite(LGBMMixin, MultiLevelEmbedderMixin, MultiOutputRegressorMixin, CiteModelABC):
    ...
