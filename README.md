# Day 1
17/09/2022
focus on CITEseq, fix of lgbm with sparse data, deeper read of cite seq data docs

## Understanding

- input of cite is made of counts log1p-normalised (RNA levels), it is really sparse
(78%), some columns are even made of just zeros
- target of cite is made by dsb-normalised protein levels, which means the log count of proteins
gaussian standardised by the mean level of an empty droplets 
i.e. $\frac{\log(x_i) - \mu_{empty,i}}{\sigma_{empty,i}}$
- the problem to be solved finally is influenced by the split of the 
train/public_test/private_test, i.e.  the private test is just for day 7, for all donors, 
while the leaderboard is made out of public test which rely on one different (from the train)
donor, so we expect the final rank to change. More specifically, the task tested in the public
does not require some time dependent forecasting, while the private does.


## Ideas and Todos:

- asses distribution of the inputs, with and without zeros
- assess svd singular values behaviour
- consider, by statistical investigation (e.g. Gamma distribution), different objective function
for LGBM, more generally how the target values do distribute
- train an encoder-decoder model to embed input feature after more conservative SVD
    - basic encoder decoder
    - multi task fashion: encoder decoder + multi-output regressor
- verify multi policy to hid cell_type, in case it is hidden only for public test set, we can
can leverage it in a multitask setting