from scmm.problems.cite.concrete import LGBMwMultilevelEmbedderCite
from scmm.problems.cite.configurations.lgbm_w_autoencoder_deep import configuration, model_label
from scmm.utils.log import setup_logging

if __name__ == "__main__":
    setup_logging("DEBUG")

    model_wrapper = LGBMwMultilevelEmbedderCite(configuration=configuration, label=model_label)
    model_wrapper.full_pipeline(save_model=True)



