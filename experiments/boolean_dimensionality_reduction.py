from single_cell_multimodal_core.data_handling import load_sparse

train_cite_input = load_sparse(split="train", problem="cite", type="inputs")

train_cite_input_b = train_cite_input > 0
list_of_sparse_row_b = [e for e in train_cite_input_b]
list_of_tuple_sparse_row_b = [tuple(e.indices) for e in train_cite_input_b]
set_of_sparse_row_b = set(list_of_tuple_sparse_row_b)
len(set_of_sparse_row_b)
