from scmm.models.cite.concrete import LGBMwSVDAutoEncoderCite
from scmm.models.cite.configurations.lgbm_w_autoencoder import configuration, model_label
from scmm.utils.log import setup_logging

if __name__ == "__main__":
    setup_logging("DEBUG")
    model_wrapper = LGBMwSVDAutoEncoderCite(configuration=configuration, label=model_label)

    model_wrapper.full_pipeline(save_model=True)
