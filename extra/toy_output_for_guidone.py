from scmm.problems.cite.concrete import LGBMwMultilevelEmbedderCite
from scmm.problems.cite.configurations.lgbm_w_autoencoder_small import configuration, model_label, model_class
from scmm.utils.log import setup_logging

if __name__ == "__main__":
    setup_logging("DEBUG")

    # from scmm.problems.cite.configurations.lgbm_w_autoencoder_small import configuration, model_label, model_class
    # model_wrapper = model_class(configuration=configuration, label=model_label)
    # model_wrapper.fit_and_predict_problem()

    from scmm.problems.multiome.configurations.svd_ridge import configuration, model_label, model_class

    model_wrapper = model_class(configuration=configuration, label=model_label)
    model_wrapper.fit_and_predict_problem()
