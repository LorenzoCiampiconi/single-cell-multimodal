def check_nn_embedder_params(embedder_params):
    for embedder_config in embedder_params["embedders_config"]:
        if "model_params" in embedder_config[1] and "shrinking_factors" in embedder_config[1]["model_params"]:
            input_dim = embedder_config[1]["input_dim"]
            output_dim = embedder_config[1]["output_dim"]

            final_dim = input_dim
            for factor in embedder_config[1]["model_params"]["shrinking_factors"]:
                final_dim = final_dim // factor

            assert final_dim == output_dim
