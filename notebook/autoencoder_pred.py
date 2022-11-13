#%% imports

from scmm.models.embedding.autoencoder.full.concrete.multitask import MultiTaskEncoderEmbedder
from scmm.problems.cite.configurations.lgbm_w_supervised_autoencoder_deep import embedder_params
from scmm.utils.data_handling import load_sparse
import joblib

#%% load

model = MultiTaskEncoderEmbedder(**embedder_params["embedders_config"][1][1])
model.load_model(
    "log/tensorboard/supervised_autoencoder_cite/version_0/checkpoints/epoch=19-step=11100.ckpt",
    **model.build_params()
)

#%% data

input = load_sparse(split="test", problem="cite", type="inputs")

svd = joblib.load("cache/svd/embedder_cite_input_dim-None_output_dim-2048_seed-0.t-svd")
red_input = svd.svd.transform(input)

#%% predict

out = model.predict(red_input)

# %%
