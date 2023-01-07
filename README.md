# single-cell-multimodal

This is a repository of model experiments for the Multimodal Single-Cell Integration kaggle competition.

The current developed pipeline is made of two abstract flexible models, 
one for each of the two problems, Multiome and Citeseq, that can be built up, trained and tuned with different options.

CiteSeq:
 - A Truncated SVD layer. This layer is mandatory to make the problem feasible (original input space is ...)
 - An optional multitask supervised embedding layer. The task employable are
  - autoenconding of the reduced input space
  - original regression problem
  - prediction of cell labelling available in the metadata (not yet developed)
 - A multioutput sklearn wrapper that can be filled individual regressor:
  - currently the model employ y lgbm regressor
 - Instead of the multioutput a linear multiregression layer can be employed, this is not effective in case there's no supervision in the extraction of the embedding

Multiome:
 - Multiome model share the same input encoding possibilites of the Citeseq one
 - on the output one can apply
  - a ridge multioutput regression
  - a lgbm that make prediction on a shrinked subspaced by mean of truncated-svd. The prediction is then projected on the full space by the inverse SVD.
