#%% imports

# %load_ext autoreload
# %autoreload 2

import joblib
import pandas as pd

from scmm.models.embedding.autoencoder.full.concrete.multitask import MultiTaskEncoderEmbedder
from scmm.problems.cite.configurations.lgbm_w_supervised_autoencoder_deep import embedder_params
from scmm.utils.appdirs import app_static_dir
from scmm.utils.data_handling import load_sparse

# %% setup

cite_exp = "pearson_mse"
# multiome_exp = "baseline"
multiome_exp = "cite_lgbm_w_4lrs-deep_supervised_autoencoder_dim-2048-128_20221107-0043_submission"

# %% load

model = MultiTaskEncoderEmbedder(**embedder_params["embedders_config"][1][1])
model.load_model(
    "log/tensorboard/supervised_autoencoder_cite/version_0/checkpoints/epoch=19-step=11100.ckpt", **model.build_params()
)

# %% data

input = load_sparse(split="test", problem="cite", type="inputs")

svd = joblib.load("cache/svd/embedder_cite_input_dim-None_output_dim-2048_seed-0.t-svd")
red_input = svd.svd.transform(input)

# %% predict

_, test_output = model.predict(red_input)

# %%  generate submission cite

df = pd.Series(test_output.ravel(), name="target").to_frame()
df["target"].iloc[0:7476] = 0
df.index.name = "row_id"
# df.to_csv(app_static_dir("out") / f"{cite_exp}.csv")

# %% combine multiome

out = pd.read_csv(f"out/{multiome_exp}.csv", index_col=0)
# df = pd.read_csv(f"out/{cite_exp}.csv", index_col=0)

out.loc[df.index, "target"] = df["target"]
out.to_csv(app_static_dir("out") / "huber_mse.csv")
