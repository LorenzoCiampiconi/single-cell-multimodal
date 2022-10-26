'''
MAIN LGBM + AUTOENCODER
from scmm.models.cite.concrete import LGBMwMultilevelEmbedderCite
from scmm.models.cite.configurations.lgbm_w_autoencoder_small import configuration, model_label
from scmm.utils.log import setup_logging

if __name__ == "__main__":
    setup_logging("DEBUG")
    model_wrapper = LGBMwMultilevelEmbedderCite(configuration=configuration, label=model_label)

    model_wrapper.full_pipeline(save_model=True)
'''

from scmm.models.cite.concrete import RFCite
from scmm.models.cite.configurations.rf import configuration,model_label
from scmm.utils.log import setup_logging

if __name__ == "__main__":
    setup_logging("DEBUG")
    model_wrapper = RFCite(configuration=configuration, label=model_label)
    model_wrapper.full_pipeline(save_model=True)
