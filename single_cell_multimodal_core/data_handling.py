import pandas as pd
import numpy as np
import scipy
import scipy.sparse

from single_cell_multimodal_core.utils.appdirs import app_static_dir


"""
this script generates (for each file):
- One "xxx_values.sparse" file that can be loaded with scipy.sparse.load_npz and contains all the values 
  of the corresponding dataframe (i.e. the result of df.values in a sparse format)
- One "xxx_idxcol.npz" file that can be loaded with np.load and contains the values of the index and the 
  columns of the corresponding dataframe (i.e the results of df.index and df.columns)
"""


def load_sparse(split="train", problem="cite", type="inputs") -> scipy.sparse.csr.csr_matrix:
    return scipy.sparse.load_npz(app_static_dir("DATA") / f"{split}_{problem}_{type}_values.sparse.npz")


def convert_to_parquet(filename, out_filename):
    df = pd.read_csv(filename)
    df.to_parquet(out_filename + ".parquet")


def convert_h5_to_sparse_csr(filename, out_filename, chunksize=2500):
    start = 0
    total_rows = 0

    sparse_chunks_data_list = []
    chunks_index_list = []
    columns_name = None

    filename = app_static_dir("DATA") / filename
    out_filename = app_static_dir("DATA") / out_filename

    while True:
        df_chunk = pd.read_hdf(filename, start=start, stop=start + chunksize)
        if len(df_chunk) == 0:
            break
        chunk_data_as_sparse = scipy.sparse.csr_matrix(df_chunk.to_numpy())
        sparse_chunks_data_list.append(chunk_data_as_sparse)
        chunks_index_list.append(df_chunk.index.to_numpy())

        if columns_name is None:
            columns_name = df_chunk.columns.to_numpy()
        else:
            assert np.all(columns_name == df_chunk.columns.to_numpy())

        total_rows += len(df_chunk)
        print(total_rows)
        if len(df_chunk) < chunksize:
            del df_chunk
            break
        del df_chunk
        start += chunksize

    all_data_sparse = scipy.sparse.vstack(sparse_chunks_data_list)
    del sparse_chunks_data_list

    all_indices = np.hstack(chunks_index_list)

    scipy.sparse.save_npz(out_filename.parent / (out_filename.name + "_values.sparse"), all_data_sparse)
    np.savez(out_filename.parent / (out_filename.name + "_idxcol.npz"), index=all_indices, columns=columns_name)


def convert_source_file_to_sparse():
    convert_h5_to_sparse_csr("train_multi_targets.h5", "train_multi_targets")
    convert_h5_to_sparse_csr("train_multi_inputs.h5", "train_multi_inputs")
    convert_h5_to_sparse_csr("train_cite_targets.h5", "train_cite_targets")
    convert_h5_to_sparse_csr("train_cite_inputs.h5", "train_cite_inputs")
    convert_h5_to_sparse_csr("test_multi_inputs.h5", "test_multi_inputs")
    convert_h5_to_sparse_csr("test_cite_inputs.h5", "test_cite_inputs")


def convert_source_csv_to_parquet():
    convert_to_parquet("metadata.csv", "metadata")
    convert_to_parquet("evaluation_ids.csv", "evaluation")
    convert_to_parquet("sample_submission.csv", "sample_submission")


if __name__ == "__main__":
    convert_source_file_to_sparse()
