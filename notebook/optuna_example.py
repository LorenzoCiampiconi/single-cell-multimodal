Optuna: A
hyperparameter
optimization
framework
In
This
Kernel
I
will
use
the
amazing
Optuna
to
find
the
best
hyparameters
of
LGBM
So, Optuna is an
automatic
hyperparameter
optimization
software
framework, particularly
designed
for machine learning.It features an imperative, define-by-run style user API.The code written with Optuna enjoys high modularity, and the user of Optuna can dynamically construct the search spaces for the hyperparameters.

To
learn
more
about
Optuna
check
this
link
Basic
Concepts
So, We
use
the
terms
study and trial as follows:

Study: optimization
based
on
an
objective
function
Trial: a
single
execution
of
the
objective
function
# !pip install optuna
import optuna
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/tabular-playground-series-feb-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-feb-2021/test.csv')
sub = pd.read_csv('../input/tabular-playground-series-feb-2021/sample_submission.csv')
train.head()
id
cat0
cat1
cat2
cat3
cat4
cat5
cat6
cat7
cat8...cont5
cont6
cont7
cont8
cont9
cont10
cont11
cont12
cont13
target
0
1
A
B
A
A
B
D
A
E
C...
0.881122
0.421650
0.741413
0.895799
0.802461
0.724417
0.701915
0.877618
0.719903
6.994023
1
2
B
A
A
A
B
B
A
E
A...
0.440011
0.346230
0.278495
0.593413
0.546056
0.613252
0.741289
0.326679
0.808464
8.071256
2
3
A
A
A
C
B
D
A
B
C...
0.914155
0.369602
0.832564
0.865620
0.825251
0.264104
0.695561
0.869133
0.828352
5.760456
3
4
A
A
A
C
B
D
A
E
G...
0.934138
0.578930
0.407313
0.868099
0.794402
0.494269
0.698125
0.809799
0.614766
7.806457
4
6
A
B
A
A
B
B
A
E
C...
0.382600
0.705940
0.325193
0.440967
0.462146
0.724447
0.683073
0.343457
0.297743
6.868974
5
rows × 26
columns

categorical_cols = ['cat' + str(i) for i in range(10)]
continous_cols = ['cont' + str(i) for i in range(14)]
Encode
categorical
features
for e in categorical_cols:
    le = LabelEncoder()
    train[e] = le.fit_transform(train[e])
    test[e] = le.transform(test[e])
data = train[categorical_cols + continous_cols]
target = train['target']
data.head()
cat0
cat1
cat2
cat3
cat4
cat5
cat6
cat7
cat8
cat9...cont4
cont5
cont6
cont7
cont8
cont9
cont10
cont11
cont12
cont13
0
0
1
0
0
1
3
0
4
2
8...
0.281421
0.881122
0.421650
0.741413
0.895799
0.802461
0.724417
0.701915
0.877618
0.719903
1
1
0
0
0
1
1
0
4
0
5...
0.282354
0.440011
0.346230
0.278495
0.593413
0.546056
0.613252
0.741289
0.326679
0.808464
2
0
0
0
2
1
3
0
1
2
13...
0.293756
0.914155
0.369602
0.832564
0.865620
0.825251
0.264104
0.695561
0.869133
0.828352
3
0
0
0
2
1
3
0
4
6
10...
0.769785
0.934138
0.578930
0.407313
0.868099
0.794402
0.494269
0.698125
0.809799
0.614766
4
0
1
0
0
1
1
0
4
2
5...
0.279105
0.382600
0.705940
0.325193
0.440967
0.462146
0.724447
0.683073
0.343457
0.297743
5
rows × 24
columns

Let
's build our optimization function using optuna
This
function
uses
LGBMRegressor
model, takes
the
data
the
target
trial(How
many
executions
we
will
do)
#### and returns
RMSE(Root
Mean
Squared
Rrror)
Notes:
Note
that
I
used
some
LGBMRegressor
hyperparameters
from LGBM official

site.
So if you
like
to
add
more
parameters or change
them, check
this
link
Also
I
used
early_stopping_rounds
to
avoid
overfiting


def objective(trial, data=data, target=target):
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=42)
    param = {
        'metric': 'rmse',
        'random_state': 48,
        'n_estimators': 20000,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]),
        'max_depth': trial.suggest_categorical('max_depth', [10, 20, 100]),
        'num_leaves': trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth': trial.suggest_int('min_data_per_groups', 1, 100)
    }
    model = LGBMRegressor(**param)

    model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100, verbose=False)

    preds = model.predict(test_x)

    rmse = mean_squared_error(test_y, preds, squared=False)

    return rmse


All
thing is ready
So
let
's start 🏄‍
Note
that
the
objective
of
our
fuction is to
minimize
the
RMSE
that
's why I set direction='
minimize
'
you
can
vary
n_trials(number
of
executions)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
[I 2021 - 02 - 04 13: 48:01, 587] A
new
study
created in memory
with name: no - name - 900040
c1 - fe9b - 4
a95 - 8e00 - bb57df00e6b1
[I 2021 - 02 - 04 13: 49:11, 592] Trial
0
finished
with value: 0.8426000693073501 and parameters: {'reg_alpha': 5.8584324267071235, 'reg_lambda': 4.533540617965413,
                                                'colsample_bytree': 0.3, 'subsample': 0.6, 'learning_rate': 0.008,
                                                'max_depth': 10, 'num_leaves': 789, 'min_child_samples': 260,
                                                'min_data_per_groups': 77}.Best is trial
0
with value: 0.8426000693073501.
[I 2021 - 02 - 04 13: 49:52, 534] Trial
1
finished
with value: 0.843073554800439 and parameters: {'reg_alpha': 0.00274228231268007, 'reg_lambda': 4.406271684675252,
                                               'colsample_bytree': 0.3, 'subsample': 1.0, 'learning_rate': 0.017,
                                               'max_depth': 20, 'num_leaves': 9, 'min_child_samples': 215,
                                               'min_data_per_groups': 30}.Best is trial
0
with value: 0.8426000693073501.
[I 2021 - 02 - 04 13: 50:40, 370] Trial
2
finished
with value: 0.8442185103864116 and parameters: {'reg_alpha': 0.7908445310685409, 'reg_lambda': 0.013316031057346118,
                                                'colsample_bytree': 0.7, 'subsample': 1.0, 'learning_rate': 0.01,
                                                'max_depth': 10, 'num_leaves': 830, 'min_child_samples': 202,
                                                'min_data_per_groups': 85}.Best is trial
0
with value: 0.8426000693073501.
[I 2021 - 02 - 04 13: 51:41, 259] Trial
3
finished
with value: 0.8445625743952987 and parameters: {'reg_alpha': 0.0029232941830314406, 'reg_lambda': 0.03445471690722277,
                                                'colsample_bytree': 0.8, 'subsample': 0.6, 'learning_rate': 0.008,
                                                'max_depth': 100, 'num_leaves': 253, 'min_child_samples': 70,
                                                'min_data_per_groups': 52}.Best is trial
0
with value: 0.8426000693073501.
[I 2021 - 02 - 04 13: 52:0
9, 01
8] Trial
4
finished
with value: 0.844662557229898 and parameters: {'reg_alpha': 0.030098531302401626, 'reg_lambda': 0.06026060098384653,
                                               'colsample_bytree': 1.0, 'subsample': 0.5, 'learning_rate': 0.017,
                                               'max_depth': 10, 'num_leaves': 66, 'min_child_samples': 101,
                                               'min_data_per_groups': 57}.Best is trial
0
with value: 0.8426000693073501.
[I 2021 - 02 - 04 13: 52:44, 818] Trial
5
finished
with value: 0.8434577561121785 and parameters: {'reg_alpha': 0.006976002650530168, 'reg_lambda': 6.923023607947745,
                                                'colsample_bytree': 0.8, 'subsample': 0.6, 'learning_rate': 0.02,
                                                'max_depth': 100, 'num_leaves': 13, 'min_child_samples': 172,
                                                'min_data_per_groups': 1}.Best is trial
0
with value: 0.8426000693073501.
[I 2021 - 02 - 04 13: 53:15, 262] Trial
6
finished
with value: 0.8453350115943181 and parameters: {'reg_alpha': 0.06851766266215795, 'reg_lambda': 0.25333817371322764,
                                                'colsample_bytree': 0.9, 'subsample': 0.7, 'learning_rate': 0.017,
                                                'max_depth': 10, 'num_leaves': 704, 'min_child_samples': 178,
                                                'min_data_per_groups': 5}.Best is trial
0
with value: 0.8426000693073501.
[I 2021 - 02 - 04 13: 55:13, 006] Trial
7
finished
with value: 0.8471608390125905 and parameters: {'reg_alpha': 1.0596513797078375, 'reg_lambda': 0.11639351046976142,
                                                'colsample_bytree': 0.9, 'subsample': 0.6, 'learning_rate': 0.006,
                                                'max_depth': 20, 'num_leaves': 987, 'min_child_samples': 144,
                                                'min_data_per_groups': 85}.Best is trial
0
with value: 0.8426000693073501.
[I 2021 - 02 - 04 13: 57:05, 517] Trial
8
finished
with value: 0.8443108807613081 and parameters: {'reg_alpha': 0.0017946563740990793, 'reg_lambda': 0.4264736778947381,
                                                'colsample_bytree': 0.4, 'subsample': 0.4, 'learning_rate': 0.006,
                                                'max_depth': 100, 'num_leaves': 823, 'min_child_samples': 4,
                                                'min_data_per_groups': 16}.Best is trial
0
with value: 0.8426000693073501.
[I 2021 - 02 - 04 13: 57:44, 993] Trial
9
finished
with value: 0.8456466892585504 and parameters: {'reg_alpha': 6.479361268328116, 'reg_lambda': 0.06547577516401966,
                                                'colsample_bytree': 0.7, 'subsample': 0.5, 'learning_rate': 0.02,
                                                'max_depth': 20, 'num_leaves': 661, 'min_child_samples': 33,
                                                'min_data_per_groups': 74}.Best is trial
0
with value: 0.8426000693073501.
[I 2021 - 02 - 04 13: 58:49, 218] Trial
10
finished
with value: 0.8424445761667749 and parameters: {'reg_alpha': 6.80122717101901, 'reg_lambda': 0.0014728170495711317,
                                                'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008,
                                                'max_depth': 10, 'num_leaves': 446, 'min_child_samples': 298,
                                                'min_data_per_groups': 95}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 13: 59:54, 827] Trial
11
finished
with value: 0.8424998830513394 and parameters: {'reg_alpha': 7.226724835337076, 'reg_lambda': 0.00175742058021985,
                                                'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008,
                                                'max_depth': 10, 'num_leaves': 461, 'min_child_samples': 299,
                                                'min_data_per_groups': 98}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 00:33, 592] Trial
12
finished
with value: 0.8433142450443933 and parameters: {'reg_alpha': 1.0169355987349664, 'reg_lambda': 0.001059865527741798,
                                                'colsample_bytree': 0.5, 'subsample': 0.8, 'learning_rate': 0.014,
                                                'max_depth': 10, 'num_leaves': 452, 'min_child_samples': 299,
                                                'min_data_per_groups': 92}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 01:36, 773] Trial
13
finished
with value: 0.8435778761396061 and parameters: {'reg_alpha': 9.509434013314971, 'reg_lambda': 0.001009199399387012,
                                                'colsample_bytree': 0.6, 'subsample': 0.8, 'learning_rate': 0.008,
                                                'max_depth': 10, 'num_leaves': 425, 'min_child_samples': 292,
                                                'min_data_per_groups': 99}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 02:34, 108] Trial
14
finished
with value: 0.8425758499590252 and parameters: {'reg_alpha': 2.1829776976222854, 'reg_lambda': 0.005037406209845421,
                                                'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008,
                                                'max_depth': 10, 'num_leaves': 306, 'min_child_samples': 253,
                                                'min_data_per_groups': 68}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 03:31, 131] Trial
15
finished
with value: 0.842629557096957 and parameters: {'reg_alpha': 0.37168336971222177, 'reg_lambda': 0.003947875609174985,
                                               'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008,
                                               'max_depth': 10, 'num_leaves': 596, 'min_child_samples': 259,
                                               'min_data_per_groups': 100}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 04:25, 786] Trial
16
finished
with value: 0.8425694576332446 and parameters: {'reg_alpha': 0.25137273187681103, 'reg_lambda': 0.003115453845451418,
                                                'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008,
                                                'max_depth': 10, 'num_leaves': 280, 'min_child_samples': 294,
                                                'min_data_per_groups': 34}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 04:58, 701] Trial
17
finished
with value: 0.8436833208209615 and parameters: {'reg_alpha': 2.936475375335593, 'reg_lambda': 0.0011387212509536226,
                                                'colsample_bytree': 0.6, 'subsample': 0.8, 'learning_rate': 0.014,
                                                'max_depth': 10, 'num_leaves': 524, 'min_child_samples': 232,
                                                'min_data_per_groups': 64}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 05:59, 765] Trial
18
finished
with value: 0.8431356975056378 and parameters: {'reg_alpha': 9.99864829736277, 'reg_lambda': 0.013335591458145188,
                                                'colsample_bytree': 0.5, 'subsample': 0.4, 'learning_rate': 0.01,
                                                'max_depth': 10, 'num_leaves': 158, 'min_child_samples': 279,
                                                'min_data_per_groups': 100}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 07:15, 135] Trial
19
finished
with value: 0.8457373882948379 and parameters: {'reg_alpha': 2.8884703242465832, 'reg_lambda': 1.369867729837633,
                                                'colsample_bytree': 1.0, 'subsample': 0.7, 'learning_rate': 0.008,
                                                'max_depth': 10, 'num_leaves': 385, 'min_child_samples': 120,
                                                'min_data_per_groups': 39}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 0
8: 25, 878] Trial
20
finished
with value: 0.8436208642747948 and parameters: {'reg_alpha': 0.17141537341510762, 'reg_lambda': 0.011375901653880562,
                                                'colsample_bytree': 0.4, 'subsample': 0.8, 'learning_rate': 0.008,
                                                'max_depth': 100, 'num_leaves': 543, 'min_child_samples': 230,
                                                'min_data_per_groups': 87}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 0
9: 26, 526] Trial
21
finished
with value: 0.8425608787561133 and parameters: {'reg_alpha': 0.022871632568076106, 'reg_lambda': 0.0025632588591765,
                                                'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008,
                                                'max_depth': 10, 'num_leaves': 243, 'min_child_samples': 299,
                                                'min_data_per_groups': 23}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 10:17, 719] Trial
22
finished
with value: 0.8425298983782161 and parameters: {'reg_alpha': 0.01810452202076926, 'reg_lambda': 0.002301750459569976,
                                                'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008,
                                                'max_depth': 10, 'num_leaves': 159, 'min_child_samples': 298,
                                                'min_data_per_groups': 15}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 11:15, 951] Trial
23
finished
with value: 0.8424820562339183 and parameters: {'reg_alpha': 0.013348449074954607, 'reg_lambda': 0.0017421208701366864,
                                                'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008,
                                                'max_depth': 10, 'num_leaves': 145, 'min_child_samples': 271,
                                                'min_data_per_groups': 47}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 12:11, 710] Trial
24
finished
with value: 0.8425760130822344 and parameters: {'reg_alpha': 0.006177696286746485, 'reg_lambda': 0.007939112374524607,
                                                'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008,
                                                'max_depth': 10, 'num_leaves': 350, 'min_child_samples': 270,
                                                'min_data_per_groups': 44}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 13:02, 237] Trial
25
finished
with value: 0.8425043107377422 and parameters: {'reg_alpha': 0.11137446333243872, 'reg_lambda': 0.001549225746109588,
                                                'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008,
                                                'max_depth': 20, 'num_leaves': 134, 'min_child_samples': 243,
                                                'min_data_per_groups': 45}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 14:27, 932] Trial
26
finished
with value: 0.8426346309863528 and parameters: {'reg_alpha': 0.0010225924005903637, 'reg_lambda': 0.026234642952996306,
                                                'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.006,
                                                'max_depth': 10, 'num_leaves': 481, 'min_child_samples': 199,
                                                'min_data_per_groups': 60}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 14:59, 951] Trial
27
finished
with value: 0.8428215207397899 and parameters: {'reg_alpha': 0.04203648263149102, 'reg_lambda': 0.0061322832686678125,
                                                'colsample_bytree': 0.3, 'subsample': 0.5, 'learning_rate': 0.014,
                                                'max_depth': 10, 'num_leaves': 618, 'min_child_samples': 276,
                                                'min_data_per_groups': 71}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 15:22, 332] Trial
28
finished
with value: 0.8427762954271026 and parameters: {'reg_alpha': 0.011776051419119114, 'reg_lambda': 0.001703531888950542,
                                                'colsample_bytree': 0.3, 'subsample': 1.0, 'learning_rate': 0.02,
                                                'max_depth': 10, 'num_leaves': 370, 'min_child_samples': 278,
                                                'min_data_per_groups': 93}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 16:13, 326] Trial
29
finished
with value: 0.8437319245562795 and parameters: {'reg_alpha': 4.244505608137845, 'reg_lambda': 0.021907927383566117,
                                                'colsample_bytree': 0.6, 'subsample': 0.7, 'learning_rate': 0.01,
                                                'max_depth': 10, 'num_leaves': 778, 'min_child_samples': 256,
                                                'min_data_per_groups': 82}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 17:21, 177] Trial
30
finished
with value: 0.8453600942747858 and parameters: {'reg_alpha': 0.06021618529870894, 'reg_lambda': 0.0010002393782399001,
                                                'colsample_bytree': 1.0, 'subsample': 0.8, 'learning_rate': 0.008,
                                                'max_depth': 10, 'num_leaves': 947, 'min_child_samples': 236,
                                                'min_data_per_groups': 79}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 18:10, 518] Trial
31
finished
with value: 0.8425102765686904 and parameters: {'reg_alpha': 0.14952601946739738, 'reg_lambda': 0.00176654564141676,
                                                'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008,
                                                'max_depth': 20, 'num_leaves': 129, 'min_child_samples': 250,
                                                'min_data_per_groups': 46}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 19:05, 386] Trial
32
finished
with value: 0.8425612134005993 and parameters: {'reg_alpha': 0.6414842052925773, 'reg_lambda': 0.0018152706633892243,
                                                'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008,
                                                'max_depth': 20, 'num_leaves': 210, 'min_child_samples': 210,
                                                'min_data_per_groups': 46}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 19:55, 0
85] Trial
33
finished
with value: 0.8424606303650155 and parameters: {'reg_alpha': 0.005502167232253572, 'reg_lambda': 0.004532498279720812,
                                                'colsample_bytree': 0.3, 'subsample': 0.4, 'learning_rate': 0.008,
                                                'max_depth': 20, 'num_leaves': 98, 'min_child_samples': 283,
                                                'min_data_per_groups': 52}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 22:06, 922] Trial
34
finished
with value: 0.8497971419137992 and parameters: {'reg_alpha': 0.00722789950956527, 'reg_lambda': 0.003648576685747174,
                                                'colsample_bytree': 0.7, 'subsample': 0.4, 'learning_rate': 0.008,
                                                'max_depth': 20, 'num_leaves': 2, 'min_child_samples': 283,
                                                'min_data_per_groups': 33}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 23:06, 538] Trial
35
finished
with value: 0.8440018177106995 and parameters: {'reg_alpha': 0.003324527118281288, 'reg_lambda': 0.0062030919172809985,
                                                'colsample_bytree': 0.8, 'subsample': 0.4, 'learning_rate': 0.008,
                                                'max_depth': 20, 'num_leaves': 73, 'min_child_samples': 267,
                                                'min_data_per_groups': 55}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 23:32, 533] Trial
36
finished
with value: 0.8425937272855156 and parameters: {'reg_alpha': 0.0010552319536933764, 'reg_lambda': 0.00890433574965775,
                                                'colsample_bytree': 0.3, 'subsample': 0.4, 'learning_rate': 0.017,
                                                'max_depth': 20, 'num_leaves': 64, 'min_child_samples': 299,
                                                'min_data_per_groups': 93}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 24:30, 352] Trial
37
finished
with value: 0.8426384029270205 and parameters: {'reg_alpha': 0.012162291826382432, 'reg_lambda': 0.0031702312717925953,
                                                'colsample_bytree': 0.3, 'subsample': 1.0, 'learning_rate': 0.008,
                                                'max_depth': 100, 'num_leaves': 324, 'min_child_samples': 221,
                                                'min_data_per_groups': 61}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 24:57, 958] Trial
38
finished
with value: 0.8451289991667301 and parameters: {'reg_alpha': 0.0021794335522894066, 'reg_lambda': 0.01634235567633058,
                                                'colsample_bytree': 0.9, 'subsample': 0.6, 'learning_rate': 0.017,
                                                'max_depth': 20, 'num_leaves': 200, 'min_child_samples': 181,
                                                'min_data_per_groups': 26}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 25:49, 976] Trial
39
finished
with value: 0.8431042598878441 and parameters: {'reg_alpha': 0.004191213555872809, 'reg_lambda': 0.04375215161549017,
                                                'colsample_bytree': 0.4, 'subsample': 0.4, 'learning_rate': 0.01,
                                                'max_depth': 10, 'num_leaves': 435, 'min_child_samples': 193,
                                                'min_data_per_groups': 53}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 26:17, 310] Trial
40
finished
with value: 0.8432184909147723 and parameters: {'reg_alpha': 0.014720473058204116, 'reg_lambda': 0.0010409776590947404,
                                                'colsample_bytree': 0.3, 'subsample': 0.5, 'learning_rate': 0.02,
                                                'max_depth': 10, 'num_leaves': 726, 'min_child_samples': 71,
                                                'min_data_per_groups': 39}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 27:10, 207] Trial
41
finished
with value: 0.842553795627764 and parameters: {'reg_alpha': 1.8868496338585026, 'reg_lambda': 0.0015527938220468848,
                                               'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008,
                                               'max_depth': 20, 'num_leaves': 79, 'min_child_samples': 244,
                                               'min_data_per_groups': 49}.Best is trial
10
with value: 0.8424445761667749.
[I 2021 - 02 - 04 14: 28:05, 258] Trial
42
finished
with value: 0.8423859659530323 and parameters: {'reg_alpha': 6.147694913504962, 'reg_lambda': 0.002457826062076097,
                                                'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008,
                                                'max_depth': 20, 'num_leaves': 111, 'min_child_samples': 285,
                                                'min_data_per_groups': 39}.Best is trial
42
with value: 0.8423859659530323.
[I 2021 - 02 - 04 14: 29:16, 652] Trial
43
finished
with value: 0.8435954039110533 and parameters: {'reg_alpha': 6.479867979165064, 'reg_lambda': 0.004790255907141213,
                                                'colsample_bytree': 0.8, 'subsample': 1.0, 'learning_rate': 0.008,
                                                'max_depth': 20, 'num_leaves': 25, 'min_child_samples': 287,
                                                'min_data_per_groups': 40}.Best is trial
42
with value: 0.8423859659530323.
[I 2021 - 02 - 04 14: 30:17, 432] Trial
44
finished
with value: 0.8433882568371416 and parameters: {'reg_alpha': 6.057561702178017, 'reg_lambda': 0.002261038946857698,
                                                'colsample_bytree': 0.5, 'subsample': 0.6, 'learning_rate': 0.008,
                                                'max_depth': 20, 'num_leaves': 220, 'min_child_samples': 266,
                                                'min_data_per_groups': 20}.Best is trial
42
with value: 0.8423859659530323.
[I 2021 - 02 - 04 14: 31:31, 145] Trial
45
finished
with value: 0.8424822456806085 and parameters: {'reg_alpha': 1.409993665324953, 'reg_lambda': 0.003810932532483848,
                                                'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.006,
                                                'max_depth': 20, 'num_leaves': 105, 'min_child_samples': 300,
                                                'min_data_per_groups': 8}.Best is trial
42
with value: 0.8423859659530323.
[I 2021 - 02 - 04 14: 32:50, 881] Trial
46
finished
with value: 0.8443946986073995 and parameters: {'reg_alpha': 3.752531494364217, 'reg_lambda': 0.21161250927174843,
                                                'colsample_bytree': 0.9, 'subsample': 0.4, 'learning_rate': 0.006,
                                                'max_depth': 20, 'num_leaves': 104, 'min_child_samples': 156,
                                                'min_data_per_groups': 7}.Best is trial
42
with value: 0.8423859659530323.
[I 2021 - 02 - 04 14: 34:22, 239] Trial
47
finished
with value: 0.8435642917649732 and parameters: {'reg_alpha': 0.0016598576894783158, 'reg_lambda': 0.007354281789907943,
                                                'colsample_bytree': 0.7, 'subsample': 0.8, 'learning_rate': 0.006,
                                                'max_depth': 20, 'num_leaves': 29, 'min_child_samples': 285,
                                                'min_data_per_groups': 12}.Best is trial
42
with value: 0.8423859659530323.
[I 2021 - 02 - 04 14: 35:36, 514] Trial
48
finished
with value: 0.8424490940932478 and parameters: {'reg_alpha': 1.5554036991574203, 'reg_lambda': 0.004934685568727651,
                                                'colsample_bytree': 0.3, 'subsample': 0.7, 'learning_rate': 0.006,
                                                'max_depth': 20, 'num_leaves': 175, 'min_child_samples': 263,
                                                'min_data_per_groups': 27}.Best is trial
42
with value: 0.8423859659530323.
[I 2021 - 02 - 04 14: 37:02, 830] Trial
49
finished
with value: 0.8424110222321542 and parameters: {'reg_alpha': 0.6054657686210433, 'reg_lambda': 0.004917117336411138,
                                                'colsample_bytree': 0.3, 'subsample': 0.7, 'learning_rate': 0.006,
                                                'max_depth': 20, 'num_leaves': 180, 'min_child_samples': 264,
                                                'min_data_per_groups': 27}.Best is trial
42
with value: 0.8423859659530323.
Number
of
finished
trials: 50
Best
trial: {'reg_alpha': 6.147694913504962, 'reg_lambda': 0.002457826062076097, 'colsample_bytree': 0.3, 'subsample': 0.8,
        'learning_rate': 0.008, 'max_depth': 20, 'num_leaves': 111, 'min_child_samples': 285, 'min_data_per_groups': 39}
study.trials_dataframe()
number
value
datetime_start
datetime_complete
duration
params_colsample_bytree
params_learning_rate
params_max_depth
params_min_child_samples
params_min_data_per_groups
params_num_leaves
params_reg_alpha
params_reg_lambda
params_subsample
state
0
0
0.842600
2021 - 02 - 04
13: 48:01.590435
2021 - 02 - 04
13: 49:11.591786
0
days
00: 01:10.001351
0.3
0.008
10
260
77
789
5.858432
4.533541
0.6
COMPLETE
1
1
0.843074
2021 - 02 - 04
13: 49:11.593611
2021 - 02 - 04
13: 49:52.534127
0
days
00: 00:40.940516
0.3
0.017
20
215
30
9
0.002742
4.406272
1.0
COMPLETE
2
2
0.844219
2021 - 02 - 04
13: 49:52.536320
2021 - 02 - 04
13: 50:40.370175
0
days
00: 00:47.833855
0.7
0.010
10
202
85
830
0.790845
0.013316
1.0
COMPLETE
3
3
0.844563
2021 - 02 - 04
13: 50:40.372803
2021 - 02 - 04
13: 51:41.258567
0
days
00: 01:00.885764
0.8
0.008
100
70
52
253
0.002923
0.034455
0.6
COMPLETE
4
4
0.844663
2021 - 02 - 04
13: 51:41.261245
2021 - 02 - 04
13: 52:09.017338
0
days
00: 00:27.756093
1.0
0.017
10
101
57
66
0.030099
0.060261
0.5
COMPLETE
5
5
0.843458
2021 - 02 - 04
13: 52:09.019432
2021 - 02 - 04
13: 52:44.818050
0
days
00: 00:35.798618
0.8
0.020
100
172
1
13
0.006976
6.923024
0.6
COMPLETE
6
6
0.845335
2021 - 02 - 04
13: 52:44.819853
2021 - 02 - 04
13: 53:15.262129
0
days
00: 00:30.442276
0.9
0.017
10
178
5
704
0.068518
0.253338
0.7
COMPLETE
7
7
0.847161
2021 - 02 - 04
13: 53:15.265626
2021 - 02 - 04
13: 55:13.005385
0
days
00: 01:57.739759
0.9
0.006
20
144
85
987
1.059651
0.116394
0.6
COMPLETE
8
8
0.844311
2021 - 02 - 04
13: 55:13.007876
2021 - 02 - 04
13: 57:05.516696
0
days
00: 01:52.508820
0.4
0.006
100
4
16
823
0.001795
0.426474
0.4
COMPLETE
9
9
0.845647
2021 - 02 - 04
13: 57:05.518663
2021 - 02 - 04
13: 57:44.993260
0
days
00: 00:39.474597
0.7
0.020
20
33
74
661
6.479361
0.065476
0.5
COMPLETE
10
10
0.842445
2021 - 02 - 04
13: 57:44.996374
2021 - 02 - 04
13: 58:49.218405
0
days
00: 01:04.222031
0.3
0.008
10
298
95
446
6.801227
0.001473
0.8
COMPLETE
11
11
0.842500
2021 - 02 - 04
13: 58:49.220415
2021 - 02 - 04
13: 59:54.827367
0
days
00: 01:05.606952
0.3
0.008
10
299
98
461
7.226725
0.001757
0.8
COMPLETE
12
12
0.843314
2021 - 02 - 04
13: 59:54.829607
2021 - 02 - 04
14: 00:33.591435
0
days
00: 00:38.761828
0.5
0.014
10
299
92
452
1.016936
0.001060
0.8
COMPLETE
13
13
0.843578
2021 - 02 - 04
14: 00:33.593721
2021 - 02 - 04
14: 01:36.772636
0
days
00: 01:03.178915
0.6
0.008
10
292
99
425
9.509434
0.001009
0.8
COMPLETE
14
14
0.842576
2021 - 02 - 04
14: 01:36.774753
2021 - 02 - 04
14: 02:34.108072
0
days
00: 00:57.333319
0.3
0.008
10
253
68
306
2.182978
0.005037
0.8
COMPLETE
15
15
0.842630
2021 - 02 - 04
14: 02:34.110084
2021 - 02 - 04
14: 03:31.130395
0
days
00: 00:57.020311
0.3
0.008
10
259
100
596
0.371683
0.003948
0.8
COMPLETE
16
16
0.842569
2021 - 02 - 04
14: 03:31.132823
2021 - 02 - 04
14: 04:25.786277
0
days
00: 00:54.653454
0.3
0.008
10
294
34
280
0.251373
0.003115
0.8
COMPLETE
17
17
0.843683
2021 - 02 - 04
14: 04:25.788314
2021 - 02 - 04
14: 04:58.700579
0
days
00: 00:32.912265
0.6
0.014
10
232
64
524
2.936475
0.001139
0.8
COMPLETE
18
18
0.843136
2021 - 02 - 04
14: 04:58.702923
2021 - 02 - 04
14: 05:59.765199
0
days
00: 01:01.062276
0.5
0.010
10
279
100
158
9.998648
0.013336
0.4
COMPLETE
19
19
0.845737
2021 - 02 - 04
14: 05:59.767464
2021 - 02 - 04
14: 07:15.134455
0
days
00: 01:15.366991
1.0
0.008
10
120
39
385
2.888470
1.369868
0.7
COMPLETE
20
20
0.843621
2021 - 02 - 04
14: 07:15.136285
2021 - 02 - 04
14: 0
8: 25.877915
0
days
00: 01:10.741630
0.4
0.008
100
230
87
543
0.171415
0.011376
0.8
COMPLETE
21
21
0.842561
2021 - 02 - 04
14: 0
8: 25.880175
2021 - 02 - 04
14: 0
9: 26.526333
0
days
00: 01:00.646158
0.3
0.008
10
299
23
243
0.022872
0.002563
0.8
COMPLETE
22
22
0.842530
2021 - 02 - 04
14: 0
9: 26.528528
2021 - 02 - 04
14: 10:17.718634
0
days
00: 00:51.190106
0.3
0.008
10
298
15
159
0.018105
0.002302
0.8
COMPLETE
23
23
0.842482
2021 - 02 - 04
14: 10:17.720945
2021 - 02 - 04
14: 11:15.951003
0
days
00: 00:58.230058
0.3
0.008
10
271
47
145
0.013348
0.001742
0.8
COMPLETE
24
24
0.842576
2021 - 02 - 04
14: 11:15.952953
2021 - 02 - 04
14: 12:11.709877
0
days
00: 00:55.756924
0.3
0.008
10
270
44
350
0.006178
0.007939
0.8
COMPLETE
25
25
0.842504
2021 - 02 - 04
14: 12:11.712400
2021 - 02 - 04
14: 13:02.236994
0
days
00: 00:50.524594
0.3
0.008
20
243
45
134
0.111374
0.001549
0.8
COMPLETE
26
26
0.842635
2021 - 02 - 04
14: 13:02.238847
2021 - 02 - 04
14: 14:27.931695
0
days
00: 01:25.692848
0.3
0.006
10
199
60
481
0.001023
0.026235
0.8
COMPLETE
27
27
0.842822
2021 - 02 - 04
14: 14:27.934057
2021 - 02 - 04
14: 14:59.950804
0
days
00: 00:32.016747
0.3
0.014
10
276
71
618
0.042036
0.006132
0.5
COMPLETE
28
28
0.842776
2021 - 02 - 04
14: 14:59.953266
2021 - 02 - 04
14: 15:22.331816
0
days
00: 00:22.378550
0.3
0.020
10
278
93
370
0.011776
0.001704
1.0
COMPLETE
29
29
0.843732
2021 - 02 - 04
14: 15:22.334855
2021 - 02 - 04
14: 16:13.325716
0
days
00: 00:50.990861
0.6
0.010
10
256
82
778
4.244506
0.021908
0.7
COMPLETE
30
30
0.845360
2021 - 02 - 04
14: 16:13.327559
2021 - 02 - 04
14: 17:21.176379
0
days
00: 01:07.848820
1.0
0.008
10
236
79
947
0.060216
0.001000
0.8
COMPLETE
31
31
0.842510
2021 - 02 - 04
14: 17:21.178423
2021 - 02 - 04
14: 18:10.517819
0
days
00: 00:49.339396
0.3
0.008
20
250
46
129
0.149526
0.001767
0.8
COMPLETE
32
32
0.842561
2021 - 02 - 04
14: 18:10.520177
2021 - 02 - 04
14: 19:05.385452
0
days
00: 00:54.865275
0.3
0.008
20
210
46
210
0.641484
0.001815
0.8
COMPLETE
33
33
0.842461
2021 - 02 - 04
14: 19:05.387350
2021 - 02 - 04
14: 19:55.084852
0
days
00: 00:49.697502
0.3
0.008
20
283
52
98
0.005502
0.004532
0.4
COMPLETE
34
34
0.849797
2021 - 02 - 04
14: 19:55.087231
2021 - 02 - 04
14: 22:06.922096
0
days
00: 02:11.834865
0.7
0.008
20
283
33
2
0.007228
0.003649
0.4
COMPLETE
35
35
0.844002
2021 - 02 - 04
14: 22:06.925332
2021 - 02 - 04
14: 23:06.537826
0
days
00: 00:59.612494
0.8
0.008
20
267
55
73
0.003325
0.006203
0.4
COMPLETE
36
36
0.842594
2021 - 02 - 04
14: 23:06.539879
2021 - 02 - 04
14: 23:32.533107
0
days
00: 00:25.993228
0.3
0.017
20
299
93
64
0.001055
0.008904
0.4
COMPLETE
37
37
0.842638
2021 - 02 - 04
14: 23:32.537253
2021 - 02 - 04
14: 24:30.351685
0
days
00: 00:57.814432
0.3
0.008
100
221
61
324
0.012162
0.003170
1.0
COMPLETE
38
38
0.845129
2021 - 02 - 04
14: 24:30.353624
2021 - 02 - 04
14: 24:57.958046
0
days
00: 00:27.604422
0.9
0.017
20
181
26
200
0.002179
0.016342
0.6
COMPLETE
39
39
0.843104
2021 - 02 - 04
14: 24:57.960299
2021 - 02 - 04
14: 25:49.976143
0
days
00: 00:52.015844
0.4
0.010
10
193
53
435
0.004191
0.043752
0.4
COMPLETE
40
40
0.843218
2021 - 02 - 04
14: 25:49.978062
2021 - 02 - 04
14: 26:17.309380
0
days
00: 00:27.331318
0.3
0.020
10
71
39
726
0.014720
0.001041
0.5
COMPLETE
41
41
0.842554
2021 - 02 - 04
14: 26:17.311454
2021 - 02 - 04
14: 27:10.206429
0
days
00: 00:52.894975
0.3
0.008
20
244
49
79
1.886850
0.001553
0.8
COMPLETE
42
42
0.842386
2021 - 02 - 04
14: 27:10.208376
2021 - 02 - 04
14: 28:05.258114
0
days
00: 00:55.049738
0.3
0.008
20
285
39
111
6.147695
0.002458
0.8
COMPLETE
43
43
0.843595
2021 - 02 - 04
14: 28:05.260060
2021 - 02 - 04
14: 29:16.651601
0
days
00: 01:11.391541
0.8
0.008
20
287
40
25
6.479868
0.004790
1.0
COMPLETE
44
44
0.843388
2021 - 02 - 04
14: 29:16.653848
2021 - 02 - 04
14: 30:17.431860
0
days
00: 01:00.778012
0.5
0.008
20
266
20
220
6.057562
0.002261
0.6
COMPLETE
45
45
0.842482
2021 - 02 - 04
14: 30:17.433920
2021 - 02 - 04
14: 31:31.144614
0
days
00: 01:13.710694
0.3
0.006
20
300
8
105
1.409994
0.003811
0.8
COMPLETE
46
46
0.844395
2021 - 02 - 04
14: 31:31.146872
2021 - 02 - 04
14: 32:50.880540
0
days
00: 01:19.733668
0.9
0.006
20
156
7
104
3.752531
0.211613
0.4
COMPLETE
47
47
0.843564
2021 - 02 - 04
14: 32:50.883161
2021 - 02 - 04
14: 34:22.238404
0
days
00: 01:31.355243
0.7
0.006
20
285
12
29
0.001660
0.007354
0.8
COMPLETE
48
48
0.842449
2021 - 02 - 04
14: 34:22.240417
2021 - 02 - 04
14: 35:36.513428
0
days
00: 01:14.273011
0.3
0.006
20
263
27
175
1.555404
0.004935
0.7
COMPLETE
49
49
0.842411
2021 - 02 - 04
14: 35:36.515519
2021 - 02 - 04
14: 37:02.829705
0
days
00: 01:26.314186
0.3
0.006
20
264
27
180
0.605466
0.004917
0.7
COMPLETE
Let
's do some Quick Visualization for Hyperparameter Optimization Analysis
Optuna
provides
various
visualization
features in optuna.visualization
to
analyze
optimization
results
visually
# plot_optimization_histor: shows the scores from all trials as well as the best score so far at each point.
optuna.visualization.plot_optimization_history(study)
# plot_parallel_coordinate: interactively visualizes the hyperparameters and scores
optuna.visualization.plot_parallel_coordinate(study)
'''plot_slice: shows the evolution of the search. You can see where in the hyperparameter space your search
went and which parts of the space were explored more.'''
optuna.visualization.plot_slice(study)
# plot_contour: plots parameter interactions on an interactive chart. You can choose which hyperparameters you would like to explore.
optuna.visualization.plot_contour(study, params=['num_leaves',
                                                 'max_depth',
                                                 'subsample',
                                                 'learning_rate',
                                                 'subsample'])
# Visualize parameter importances.
optuna.visualization.plot_param_importances(study)
# Visualize empirical distribution function
optuna.visualization.plot_edf(study)
Let
's create an LGBMRegressor model with the best hyperparameters
params = study.best_params
params['random_state'] = 48
params['n_estimators'] = 20000
params['metric'] = 'rmse'
# I changed min_data_per_groups to cat_smooth beacuse when I used LGBM params in optuna I named cat_smooth
# as min_data_per_groups (there is no parameter named min_data_per_groups in LGBM !!!)
params['cat_smooth'] = params.pop('min_data_per_groups')
params
{'reg_alpha': 6.147694913504962,
 'reg_lambda': 0.002457826062076097,
 'colsample_bytree': 0.3,
 'subsample': 0.8,
 'learning_rate': 0.008,
 'max_depth': 20,
 'num_leaves': 111,
 'min_child_samples': 285,
 'random_state': 48,
 'n_estimators': 20000,
 'metric': 'rmse',
 'cat_smooth': 39}
columns = categorical_cols + continous_cols
preds = np.zeros(test.shape[0])
kf = KFold(n_splits=5, random_state=48, shuffle=True)
rmse = []  # list contains rmse for each fold
n = 0
for trn_idx, test_idx in kf.split(train[columns], train['target']):
    X_tr, X_val = train[columns].iloc[trn_idx], train[columns].iloc[test_idx]
    y_tr, y_val = train['target'].iloc[trn_idx], train['target'].iloc[test_idx]
    model = LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    preds += model.predict(test[columns]) / kf.n_splits
    rmse.append(mean_squared_error(y_val, model.predict(X_val), squared=False))
    print(n + 1, rmse[n])
    n += 1
1
0.8422516588731099
2
0.8454641176530182
3
0.8435247420802582
4
0.8396584022816828
5
0.8409088145101996
np.mean(rmse)
0.8423615470796537
from optuna.integration import lightgbm as lgb

lgb.plot_importance(model, max_num_features=10, figsize=(10, 10))
plt.show()

Submission
sub['target'] = preds
sub.to_csv('submission.csv', index=False)
# Please If you find this kernel helpful, upvote it to help others see it 😊 😋Optuna: A hyperparameter optimization framework
# In This Kernel I will use the amazing Optuna to find the best hyparameters of LGBM
# So, Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It features an imperative, define-by-run style user API. The code written with Optuna enjoys high modularity, and the user of Optuna can dynamically construct the search spaces for the hyperparameters.
#
# To learn more about Optuna check this link
# Basic Concepts
# So, We use the terms study and trial as follows:
#
# Study: optimization based on an objective function
# Trial: a single execution of the objective function
# #!pip install optuna
# import optuna
# from lightgbm import LGBMRegressor
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# train = pd.read_csv('../input/tabular-playground-series-feb-2021/train.csv')
# test  = pd.read_csv('../input/tabular-playground-series-feb-2021/test.csv')
# sub = pd.read_csv('../input/tabular-playground-series-feb-2021/sample_submission.csv')
# train.head()
# id	cat0	cat1	cat2	cat3	cat4	cat5	cat6	cat7	cat8	...	cont5	cont6	cont7	cont8	cont9	cont10	cont11	cont12	cont13	target
# 0	1	A	B	A	A	B	D	A	E	C	...	0.881122	0.421650	0.741413	0.895799	0.802461	0.724417	0.701915	0.877618	0.719903	6.994023
# 1	2	B	A	A	A	B	B	A	E	A	...	0.440011	0.346230	0.278495	0.593413	0.546056	0.613252	0.741289	0.326679	0.808464	8.071256
# 2	3	A	A	A	C	B	D	A	B	C	...	0.914155	0.369602	0.832564	0.865620	0.825251	0.264104	0.695561	0.869133	0.828352	5.760456
# 3	4	A	A	A	C	B	D	A	E	G	...	0.934138	0.578930	0.407313	0.868099	0.794402	0.494269	0.698125	0.809799	0.614766	7.806457
# 4	6	A	B	A	A	B	B	A	E	C	...	0.382600	0.705940	0.325193	0.440967	0.462146	0.724447	0.683073	0.343457	0.297743	6.868974
# 5 rows × 26 columns
#
# categorical_cols=['cat'+str(i) for i in range(10)]
# continous_cols=['cont'+str(i) for i in range(14)]
# Encode categorical features
# for e in categorical_cols:
#     le = LabelEncoder()
#     train[e]=le.fit_transform(train[e])
#     test[e]=le.transform(test[e])
# data=train[categorical_cols+continous_cols]
# target=train['target']
# data.head()
# cat0	cat1	cat2	cat3	cat4	cat5	cat6	cat7	cat8	cat9	...	cont4	cont5	cont6	cont7	cont8	cont9	cont10	cont11	cont12	cont13
# 0	0	1	0	0	1	3	0	4	2	8	...	0.281421	0.881122	0.421650	0.741413	0.895799	0.802461	0.724417	0.701915	0.877618	0.719903
# 1	1	0	0	0	1	1	0	4	0	5	...	0.282354	0.440011	0.346230	0.278495	0.593413	0.546056	0.613252	0.741289	0.326679	0.808464
# 2	0	0	0	2	1	3	0	1	2	13	...	0.293756	0.914155	0.369602	0.832564	0.865620	0.825251	0.264104	0.695561	0.869133	0.828352
# 3	0	0	0	2	1	3	0	4	6	10	...	0.769785	0.934138	0.578930	0.407313	0.868099	0.794402	0.494269	0.698125	0.809799	0.614766
# 4	0	1	0	0	1	1	0	4	2	5	...	0.279105	0.382600	0.705940	0.325193	0.440967	0.462146	0.724447	0.683073	0.343457	0.297743
# 5 rows × 24 columns
#
# Let's build our optimization function using optuna
# This function uses LGBMRegressor model, takes
# the data
# the target
# trial(How many executions we will do)
# #### and returns
# RMSE(Root Mean Squared Rrror)
# Notes:
# Note that I used some LGBMRegressor hyperparameters from LGBM official site.
# So if you like to add more parameters or change them, check this link
# Also I used early_stopping_rounds to avoid overfiting
# def objective(trial,data=data,target=target):
#
#     train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2,random_state=42)
#     param = {
#         'metric': 'rmse',
#         'random_state': 48,
#         'n_estimators': 20000,
#         'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
#         'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
#         'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
#         'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
#         'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02]),
#         'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
#         'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
#         'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
#         'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
#     }
#     model = LGBMRegressor(**param)
#
#     model.fit(train_x,train_y,eval_set=[(test_x,test_y)],early_stopping_rounds=100,verbose=False)
#
#     preds = model.predict(test_x)
#
#     rmse = mean_squared_error(test_y, preds,squared=False)
#
#     return rmse
# All thing is ready So let's start 🏄‍
# Note that the objective of our fuction is to minimize the RMSE that's why I set direction='minimize'
# you can vary n_trials(number of executions)
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)
# print('Number of finished trials:', len(study.trials))
# print('Best trial:', study.best_trial.params)
# [I 2021-02-04 13:48:01,587] A new study created in memory with name: no-name-900040c1-fe9b-4a95-8e00-bb57df00e6b1
# [I 2021-02-04 13:49:11,592] Trial 0 finished with value: 0.8426000693073501 and parameters: {'reg_alpha': 5.8584324267071235, 'reg_lambda': 4.533540617965413, 'colsample_bytree': 0.3, 'subsample': 0.6, 'learning_rate': 0.008, 'max_depth': 10, 'num_leaves': 789, 'min_child_samples': 260, 'min_data_per_groups': 77}. Best is trial 0 with value: 0.8426000693073501.
# [I 2021-02-04 13:49:52,534] Trial 1 finished with value: 0.843073554800439 and parameters: {'reg_alpha': 0.00274228231268007, 'reg_lambda': 4.406271684675252, 'colsample_bytree': 0.3, 'subsample': 1.0, 'learning_rate': 0.017, 'max_depth': 20, 'num_leaves': 9, 'min_child_samples': 215, 'min_data_per_groups': 30}. Best is trial 0 with value: 0.8426000693073501.
# [I 2021-02-04 13:50:40,370] Trial 2 finished with value: 0.8442185103864116 and parameters: {'reg_alpha': 0.7908445310685409, 'reg_lambda': 0.013316031057346118, 'colsample_bytree': 0.7, 'subsample': 1.0, 'learning_rate': 0.01, 'max_depth': 10, 'num_leaves': 830, 'min_child_samples': 202, 'min_data_per_groups': 85}. Best is trial 0 with value: 0.8426000693073501.
# [I 2021-02-04 13:51:41,259] Trial 3 finished with value: 0.8445625743952987 and parameters: {'reg_alpha': 0.0029232941830314406, 'reg_lambda': 0.03445471690722277, 'colsample_bytree': 0.8, 'subsample': 0.6, 'learning_rate': 0.008, 'max_depth': 100, 'num_leaves': 253, 'min_child_samples': 70, 'min_data_per_groups': 52}. Best is trial 0 with value: 0.8426000693073501.
# [I 2021-02-04 13:52:09,018] Trial 4 finished with value: 0.844662557229898 and parameters: {'reg_alpha': 0.030098531302401626, 'reg_lambda': 0.06026060098384653, 'colsample_bytree': 1.0, 'subsample': 0.5, 'learning_rate': 0.017, 'max_depth': 10, 'num_leaves': 66, 'min_child_samples': 101, 'min_data_per_groups': 57}. Best is trial 0 with value: 0.8426000693073501.
# [I 2021-02-04 13:52:44,818] Trial 5 finished with value: 0.8434577561121785 and parameters: {'reg_alpha': 0.006976002650530168, 'reg_lambda': 6.923023607947745, 'colsample_bytree': 0.8, 'subsample': 0.6, 'learning_rate': 0.02, 'max_depth': 100, 'num_leaves': 13, 'min_child_samples': 172, 'min_data_per_groups': 1}. Best is trial 0 with value: 0.8426000693073501.
# [I 2021-02-04 13:53:15,262] Trial 6 finished with value: 0.8453350115943181 and parameters: {'reg_alpha': 0.06851766266215795, 'reg_lambda': 0.25333817371322764, 'colsample_bytree': 0.9, 'subsample': 0.7, 'learning_rate': 0.017, 'max_depth': 10, 'num_leaves': 704, 'min_child_samples': 178, 'min_data_per_groups': 5}. Best is trial 0 with value: 0.8426000693073501.
# [I 2021-02-04 13:55:13,006] Trial 7 finished with value: 0.8471608390125905 and parameters: {'reg_alpha': 1.0596513797078375, 'reg_lambda': 0.11639351046976142, 'colsample_bytree': 0.9, 'subsample': 0.6, 'learning_rate': 0.006, 'max_depth': 20, 'num_leaves': 987, 'min_child_samples': 144, 'min_data_per_groups': 85}. Best is trial 0 with value: 0.8426000693073501.
# [I 2021-02-04 13:57:05,517] Trial 8 finished with value: 0.8443108807613081 and parameters: {'reg_alpha': 0.0017946563740990793, 'reg_lambda': 0.4264736778947381, 'colsample_bytree': 0.4, 'subsample': 0.4, 'learning_rate': 0.006, 'max_depth': 100, 'num_leaves': 823, 'min_child_samples': 4, 'min_data_per_groups': 16}. Best is trial 0 with value: 0.8426000693073501.
# [I 2021-02-04 13:57:44,993] Trial 9 finished with value: 0.8456466892585504 and parameters: {'reg_alpha': 6.479361268328116, 'reg_lambda': 0.06547577516401966, 'colsample_bytree': 0.7, 'subsample': 0.5, 'learning_rate': 0.02, 'max_depth': 20, 'num_leaves': 661, 'min_child_samples': 33, 'min_data_per_groups': 74}. Best is trial 0 with value: 0.8426000693073501.
# [I 2021-02-04 13:58:49,218] Trial 10 finished with value: 0.8424445761667749 and parameters: {'reg_alpha': 6.80122717101901, 'reg_lambda': 0.0014728170495711317, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 10, 'num_leaves': 446, 'min_child_samples': 298, 'min_data_per_groups': 95}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 13:59:54,827] Trial 11 finished with value: 0.8424998830513394 and parameters: {'reg_alpha': 7.226724835337076, 'reg_lambda': 0.00175742058021985, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 10, 'num_leaves': 461, 'min_child_samples': 299, 'min_data_per_groups': 98}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:00:33,592] Trial 12 finished with value: 0.8433142450443933 and parameters: {'reg_alpha': 1.0169355987349664, 'reg_lambda': 0.001059865527741798, 'colsample_bytree': 0.5, 'subsample': 0.8, 'learning_rate': 0.014, 'max_depth': 10, 'num_leaves': 452, 'min_child_samples': 299, 'min_data_per_groups': 92}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:01:36,773] Trial 13 finished with value: 0.8435778761396061 and parameters: {'reg_alpha': 9.509434013314971, 'reg_lambda': 0.001009199399387012, 'colsample_bytree': 0.6, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 10, 'num_leaves': 425, 'min_child_samples': 292, 'min_data_per_groups': 99}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:02:34,108] Trial 14 finished with value: 0.8425758499590252 and parameters: {'reg_alpha': 2.1829776976222854, 'reg_lambda': 0.005037406209845421, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 10, 'num_leaves': 306, 'min_child_samples': 253, 'min_data_per_groups': 68}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:03:31,131] Trial 15 finished with value: 0.842629557096957 and parameters: {'reg_alpha': 0.37168336971222177, 'reg_lambda': 0.003947875609174985, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 10, 'num_leaves': 596, 'min_child_samples': 259, 'min_data_per_groups': 100}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:04:25,786] Trial 16 finished with value: 0.8425694576332446 and parameters: {'reg_alpha': 0.25137273187681103, 'reg_lambda': 0.003115453845451418, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 10, 'num_leaves': 280, 'min_child_samples': 294, 'min_data_per_groups': 34}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:04:58,701] Trial 17 finished with value: 0.8436833208209615 and parameters: {'reg_alpha': 2.936475375335593, 'reg_lambda': 0.0011387212509536226, 'colsample_bytree': 0.6, 'subsample': 0.8, 'learning_rate': 0.014, 'max_depth': 10, 'num_leaves': 524, 'min_child_samples': 232, 'min_data_per_groups': 64}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:05:59,765] Trial 18 finished with value: 0.8431356975056378 and parameters: {'reg_alpha': 9.99864829736277, 'reg_lambda': 0.013335591458145188, 'colsample_bytree': 0.5, 'subsample': 0.4, 'learning_rate': 0.01, 'max_depth': 10, 'num_leaves': 158, 'min_child_samples': 279, 'min_data_per_groups': 100}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:07:15,135] Trial 19 finished with value: 0.8457373882948379 and parameters: {'reg_alpha': 2.8884703242465832, 'reg_lambda': 1.369867729837633, 'colsample_bytree': 1.0, 'subsample': 0.7, 'learning_rate': 0.008, 'max_depth': 10, 'num_leaves': 385, 'min_child_samples': 120, 'min_data_per_groups': 39}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:08:25,878] Trial 20 finished with value: 0.8436208642747948 and parameters: {'reg_alpha': 0.17141537341510762, 'reg_lambda': 0.011375901653880562, 'colsample_bytree': 0.4, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 100, 'num_leaves': 543, 'min_child_samples': 230, 'min_data_per_groups': 87}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:09:26,526] Trial 21 finished with value: 0.8425608787561133 and parameters: {'reg_alpha': 0.022871632568076106, 'reg_lambda': 0.0025632588591765, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 10, 'num_leaves': 243, 'min_child_samples': 299, 'min_data_per_groups': 23}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:10:17,719] Trial 22 finished with value: 0.8425298983782161 and parameters: {'reg_alpha': 0.01810452202076926, 'reg_lambda': 0.002301750459569976, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 10, 'num_leaves': 159, 'min_child_samples': 298, 'min_data_per_groups': 15}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:11:15,951] Trial 23 finished with value: 0.8424820562339183 and parameters: {'reg_alpha': 0.013348449074954607, 'reg_lambda': 0.0017421208701366864, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 10, 'num_leaves': 145, 'min_child_samples': 271, 'min_data_per_groups': 47}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:12:11,710] Trial 24 finished with value: 0.8425760130822344 and parameters: {'reg_alpha': 0.006177696286746485, 'reg_lambda': 0.007939112374524607, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 10, 'num_leaves': 350, 'min_child_samples': 270, 'min_data_per_groups': 44}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:13:02,237] Trial 25 finished with value: 0.8425043107377422 and parameters: {'reg_alpha': 0.11137446333243872, 'reg_lambda': 0.001549225746109588, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 20, 'num_leaves': 134, 'min_child_samples': 243, 'min_data_per_groups': 45}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:14:27,932] Trial 26 finished with value: 0.8426346309863528 and parameters: {'reg_alpha': 0.0010225924005903637, 'reg_lambda': 0.026234642952996306, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.006, 'max_depth': 10, 'num_leaves': 481, 'min_child_samples': 199, 'min_data_per_groups': 60}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:14:59,951] Trial 27 finished with value: 0.8428215207397899 and parameters: {'reg_alpha': 0.04203648263149102, 'reg_lambda': 0.0061322832686678125, 'colsample_bytree': 0.3, 'subsample': 0.5, 'learning_rate': 0.014, 'max_depth': 10, 'num_leaves': 618, 'min_child_samples': 276, 'min_data_per_groups': 71}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:15:22,332] Trial 28 finished with value: 0.8427762954271026 and parameters: {'reg_alpha': 0.011776051419119114, 'reg_lambda': 0.001703531888950542, 'colsample_bytree': 0.3, 'subsample': 1.0, 'learning_rate': 0.02, 'max_depth': 10, 'num_leaves': 370, 'min_child_samples': 278, 'min_data_per_groups': 93}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:16:13,326] Trial 29 finished with value: 0.8437319245562795 and parameters: {'reg_alpha': 4.244505608137845, 'reg_lambda': 0.021907927383566117, 'colsample_bytree': 0.6, 'subsample': 0.7, 'learning_rate': 0.01, 'max_depth': 10, 'num_leaves': 778, 'min_child_samples': 256, 'min_data_per_groups': 82}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:17:21,177] Trial 30 finished with value: 0.8453600942747858 and parameters: {'reg_alpha': 0.06021618529870894, 'reg_lambda': 0.0010002393782399001, 'colsample_bytree': 1.0, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 10, 'num_leaves': 947, 'min_child_samples': 236, 'min_data_per_groups': 79}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:18:10,518] Trial 31 finished with value: 0.8425102765686904 and parameters: {'reg_alpha': 0.14952601946739738, 'reg_lambda': 0.00176654564141676, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 20, 'num_leaves': 129, 'min_child_samples': 250, 'min_data_per_groups': 46}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:19:05,386] Trial 32 finished with value: 0.8425612134005993 and parameters: {'reg_alpha': 0.6414842052925773, 'reg_lambda': 0.0018152706633892243, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 20, 'num_leaves': 210, 'min_child_samples': 210, 'min_data_per_groups': 46}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:19:55,085] Trial 33 finished with value: 0.8424606303650155 and parameters: {'reg_alpha': 0.005502167232253572, 'reg_lambda': 0.004532498279720812, 'colsample_bytree': 0.3, 'subsample': 0.4, 'learning_rate': 0.008, 'max_depth': 20, 'num_leaves': 98, 'min_child_samples': 283, 'min_data_per_groups': 52}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:22:06,922] Trial 34 finished with value: 0.8497971419137992 and parameters: {'reg_alpha': 0.00722789950956527, 'reg_lambda': 0.003648576685747174, 'colsample_bytree': 0.7, 'subsample': 0.4, 'learning_rate': 0.008, 'max_depth': 20, 'num_leaves': 2, 'min_child_samples': 283, 'min_data_per_groups': 33}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:23:06,538] Trial 35 finished with value: 0.8440018177106995 and parameters: {'reg_alpha': 0.003324527118281288, 'reg_lambda': 0.0062030919172809985, 'colsample_bytree': 0.8, 'subsample': 0.4, 'learning_rate': 0.008, 'max_depth': 20, 'num_leaves': 73, 'min_child_samples': 267, 'min_data_per_groups': 55}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:23:32,533] Trial 36 finished with value: 0.8425937272855156 and parameters: {'reg_alpha': 0.0010552319536933764, 'reg_lambda': 0.00890433574965775, 'colsample_bytree': 0.3, 'subsample': 0.4, 'learning_rate': 0.017, 'max_depth': 20, 'num_leaves': 64, 'min_child_samples': 299, 'min_data_per_groups': 93}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:24:30,352] Trial 37 finished with value: 0.8426384029270205 and parameters: {'reg_alpha': 0.012162291826382432, 'reg_lambda': 0.0031702312717925953, 'colsample_bytree': 0.3, 'subsample': 1.0, 'learning_rate': 0.008, 'max_depth': 100, 'num_leaves': 324, 'min_child_samples': 221, 'min_data_per_groups': 61}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:24:57,958] Trial 38 finished with value: 0.8451289991667301 and parameters: {'reg_alpha': 0.0021794335522894066, 'reg_lambda': 0.01634235567633058, 'colsample_bytree': 0.9, 'subsample': 0.6, 'learning_rate': 0.017, 'max_depth': 20, 'num_leaves': 200, 'min_child_samples': 181, 'min_data_per_groups': 26}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:25:49,976] Trial 39 finished with value: 0.8431042598878441 and parameters: {'reg_alpha': 0.004191213555872809, 'reg_lambda': 0.04375215161549017, 'colsample_bytree': 0.4, 'subsample': 0.4, 'learning_rate': 0.01, 'max_depth': 10, 'num_leaves': 435, 'min_child_samples': 193, 'min_data_per_groups': 53}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:26:17,310] Trial 40 finished with value: 0.8432184909147723 and parameters: {'reg_alpha': 0.014720473058204116, 'reg_lambda': 0.0010409776590947404, 'colsample_bytree': 0.3, 'subsample': 0.5, 'learning_rate': 0.02, 'max_depth': 10, 'num_leaves': 726, 'min_child_samples': 71, 'min_data_per_groups': 39}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:27:10,207] Trial 41 finished with value: 0.842553795627764 and parameters: {'reg_alpha': 1.8868496338585026, 'reg_lambda': 0.0015527938220468848, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 20, 'num_leaves': 79, 'min_child_samples': 244, 'min_data_per_groups': 49}. Best is trial 10 with value: 0.8424445761667749.
# [I 2021-02-04 14:28:05,258] Trial 42 finished with value: 0.8423859659530323 and parameters: {'reg_alpha': 6.147694913504962, 'reg_lambda': 0.002457826062076097, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 20, 'num_leaves': 111, 'min_child_samples': 285, 'min_data_per_groups': 39}. Best is trial 42 with value: 0.8423859659530323.
# [I 2021-02-04 14:29:16,652] Trial 43 finished with value: 0.8435954039110533 and parameters: {'reg_alpha': 6.479867979165064, 'reg_lambda': 0.004790255907141213, 'colsample_bytree': 0.8, 'subsample': 1.0, 'learning_rate': 0.008, 'max_depth': 20, 'num_leaves': 25, 'min_child_samples': 287, 'min_data_per_groups': 40}. Best is trial 42 with value: 0.8423859659530323.
# [I 2021-02-04 14:30:17,432] Trial 44 finished with value: 0.8433882568371416 and parameters: {'reg_alpha': 6.057561702178017, 'reg_lambda': 0.002261038946857698, 'colsample_bytree': 0.5, 'subsample': 0.6, 'learning_rate': 0.008, 'max_depth': 20, 'num_leaves': 220, 'min_child_samples': 266, 'min_data_per_groups': 20}. Best is trial 42 with value: 0.8423859659530323.
# [I 2021-02-04 14:31:31,145] Trial 45 finished with value: 0.8424822456806085 and parameters: {'reg_alpha': 1.409993665324953, 'reg_lambda': 0.003810932532483848, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.006, 'max_depth': 20, 'num_leaves': 105, 'min_child_samples': 300, 'min_data_per_groups': 8}. Best is trial 42 with value: 0.8423859659530323.
# [I 2021-02-04 14:32:50,881] Trial 46 finished with value: 0.8443946986073995 and parameters: {'reg_alpha': 3.752531494364217, 'reg_lambda': 0.21161250927174843, 'colsample_bytree': 0.9, 'subsample': 0.4, 'learning_rate': 0.006, 'max_depth': 20, 'num_leaves': 104, 'min_child_samples': 156, 'min_data_per_groups': 7}. Best is trial 42 with value: 0.8423859659530323.
# [I 2021-02-04 14:34:22,239] Trial 47 finished with value: 0.8435642917649732 and parameters: {'reg_alpha': 0.0016598576894783158, 'reg_lambda': 0.007354281789907943, 'colsample_bytree': 0.7, 'subsample': 0.8, 'learning_rate': 0.006, 'max_depth': 20, 'num_leaves': 29, 'min_child_samples': 285, 'min_data_per_groups': 12}. Best is trial 42 with value: 0.8423859659530323.
# [I 2021-02-04 14:35:36,514] Trial 48 finished with value: 0.8424490940932478 and parameters: {'reg_alpha': 1.5554036991574203, 'reg_lambda': 0.004934685568727651, 'colsample_bytree': 0.3, 'subsample': 0.7, 'learning_rate': 0.006, 'max_depth': 20, 'num_leaves': 175, 'min_child_samples': 263, 'min_data_per_groups': 27}. Best is trial 42 with value: 0.8423859659530323.
# [I 2021-02-04 14:37:02,830] Trial 49 finished with value: 0.8424110222321542 and parameters: {'reg_alpha': 0.6054657686210433, 'reg_lambda': 0.004917117336411138, 'colsample_bytree': 0.3, 'subsample': 0.7, 'learning_rate': 0.006, 'max_depth': 20, 'num_leaves': 180, 'min_child_samples': 264, 'min_data_per_groups': 27}. Best is trial 42 with value: 0.8423859659530323.
# Number of finished trials: 50
# Best trial: {'reg_alpha': 6.147694913504962, 'reg_lambda': 0.002457826062076097, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 20, 'num_leaves': 111, 'min_child_samples': 285, 'min_data_per_groups': 39}
# study.trials_dataframe()
# number	value	datetime_start	datetime_complete	duration	params_colsample_bytree	params_learning_rate	params_max_depth	params_min_child_samples	params_min_data_per_groups	params_num_leaves	params_reg_alpha	params_reg_lambda	params_subsample	state
# 0	0	0.842600	2021-02-04 13:48:01.590435	2021-02-04 13:49:11.591786	0 days 00:01:10.001351	0.3	0.008	10	260	77	789	5.858432	4.533541	0.6	COMPLETE
# 1	1	0.843074	2021-02-04 13:49:11.593611	2021-02-04 13:49:52.534127	0 days 00:00:40.940516	0.3	0.017	20	215	30	9	0.002742	4.406272	1.0	COMPLETE
# 2	2	0.844219	2021-02-04 13:49:52.536320	2021-02-04 13:50:40.370175	0 days 00:00:47.833855	0.7	0.010	10	202	85	830	0.790845	0.013316	1.0	COMPLETE
# 3	3	0.844563	2021-02-04 13:50:40.372803	2021-02-04 13:51:41.258567	0 days 00:01:00.885764	0.8	0.008	100	70	52	253	0.002923	0.034455	0.6	COMPLETE
# 4	4	0.844663	2021-02-04 13:51:41.261245	2021-02-04 13:52:09.017338	0 days 00:00:27.756093	1.0	0.017	10	101	57	66	0.030099	0.060261	0.5	COMPLETE
# 5	5	0.843458	2021-02-04 13:52:09.019432	2021-02-04 13:52:44.818050	0 days 00:00:35.798618	0.8	0.020	100	172	1	13	0.006976	6.923024	0.6	COMPLETE
# 6	6	0.845335	2021-02-04 13:52:44.819853	2021-02-04 13:53:15.262129	0 days 00:00:30.442276	0.9	0.017	10	178	5	704	0.068518	0.253338	0.7	COMPLETE
# 7	7	0.847161	2021-02-04 13:53:15.265626	2021-02-04 13:55:13.005385	0 days 00:01:57.739759	0.9	0.006	20	144	85	987	1.059651	0.116394	0.6	COMPLETE
# 8	8	0.844311	2021-02-04 13:55:13.007876	2021-02-04 13:57:05.516696	0 days 00:01:52.508820	0.4	0.006	100	4	16	823	0.001795	0.426474	0.4	COMPLETE
# 9	9	0.845647	2021-02-04 13:57:05.518663	2021-02-04 13:57:44.993260	0 days 00:00:39.474597	0.7	0.020	20	33	74	661	6.479361	0.065476	0.5	COMPLETE
# 10	10	0.842445	2021-02-04 13:57:44.996374	2021-02-04 13:58:49.218405	0 days 00:01:04.222031	0.3	0.008	10	298	95	446	6.801227	0.001473	0.8	COMPLETE
# 11	11	0.842500	2021-02-04 13:58:49.220415	2021-02-04 13:59:54.827367	0 days 00:01:05.606952	0.3	0.008	10	299	98	461	7.226725	0.001757	0.8	COMPLETE
# 12	12	0.843314	2021-02-04 13:59:54.829607	2021-02-04 14:00:33.591435	0 days 00:00:38.761828	0.5	0.014	10	299	92	452	1.016936	0.001060	0.8	COMPLETE
# 13	13	0.843578	2021-02-04 14:00:33.593721	2021-02-04 14:01:36.772636	0 days 00:01:03.178915	0.6	0.008	10	292	99	425	9.509434	0.001009	0.8	COMPLETE
# 14	14	0.842576	2021-02-04 14:01:36.774753	2021-02-04 14:02:34.108072	0 days 00:00:57.333319	0.3	0.008	10	253	68	306	2.182978	0.005037	0.8	COMPLETE
# 15	15	0.842630	2021-02-04 14:02:34.110084	2021-02-04 14:03:31.130395	0 days 00:00:57.020311	0.3	0.008	10	259	100	596	0.371683	0.003948	0.8	COMPLETE
# 16	16	0.842569	2021-02-04 14:03:31.132823	2021-02-04 14:04:25.786277	0 days 00:00:54.653454	0.3	0.008	10	294	34	280	0.251373	0.003115	0.8	COMPLETE
# 17	17	0.843683	2021-02-04 14:04:25.788314	2021-02-04 14:04:58.700579	0 days 00:00:32.912265	0.6	0.014	10	232	64	524	2.936475	0.001139	0.8	COMPLETE
# 18	18	0.843136	2021-02-04 14:04:58.702923	2021-02-04 14:05:59.765199	0 days 00:01:01.062276	0.5	0.010	10	279	100	158	9.998648	0.013336	0.4	COMPLETE
# 19	19	0.845737	2021-02-04 14:05:59.767464	2021-02-04 14:07:15.134455	0 days 00:01:15.366991	1.0	0.008	10	120	39	385	2.888470	1.369868	0.7	COMPLETE
# 20	20	0.843621	2021-02-04 14:07:15.136285	2021-02-04 14:08:25.877915	0 days 00:01:10.741630	0.4	0.008	100	230	87	543	0.171415	0.011376	0.8	COMPLETE
# 21	21	0.842561	2021-02-04 14:08:25.880175	2021-02-04 14:09:26.526333	0 days 00:01:00.646158	0.3	0.008	10	299	23	243	0.022872	0.002563	0.8	COMPLETE
# 22	22	0.842530	2021-02-04 14:09:26.528528	2021-02-04 14:10:17.718634	0 days 00:00:51.190106	0.3	0.008	10	298	15	159	0.018105	0.002302	0.8	COMPLETE
# 23	23	0.842482	2021-02-04 14:10:17.720945	2021-02-04 14:11:15.951003	0 days 00:00:58.230058	0.3	0.008	10	271	47	145	0.013348	0.001742	0.8	COMPLETE
# 24	24	0.842576	2021-02-04 14:11:15.952953	2021-02-04 14:12:11.709877	0 days 00:00:55.756924	0.3	0.008	10	270	44	350	0.006178	0.007939	0.8	COMPLETE
# 25	25	0.842504	2021-02-04 14:12:11.712400	2021-02-04 14:13:02.236994	0 days 00:00:50.524594	0.3	0.008	20	243	45	134	0.111374	0.001549	0.8	COMPLETE
# 26	26	0.842635	2021-02-04 14:13:02.238847	2021-02-04 14:14:27.931695	0 days 00:01:25.692848	0.3	0.006	10	199	60	481	0.001023	0.026235	0.8	COMPLETE
# 27	27	0.842822	2021-02-04 14:14:27.934057	2021-02-04 14:14:59.950804	0 days 00:00:32.016747	0.3	0.014	10	276	71	618	0.042036	0.006132	0.5	COMPLETE
# 28	28	0.842776	2021-02-04 14:14:59.953266	2021-02-04 14:15:22.331816	0 days 00:00:22.378550	0.3	0.020	10	278	93	370	0.011776	0.001704	1.0	COMPLETE
# 29	29	0.843732	2021-02-04 14:15:22.334855	2021-02-04 14:16:13.325716	0 days 00:00:50.990861	0.6	0.010	10	256	82	778	4.244506	0.021908	0.7	COMPLETE
# 30	30	0.845360	2021-02-04 14:16:13.327559	2021-02-04 14:17:21.176379	0 days 00:01:07.848820	1.0	0.008	10	236	79	947	0.060216	0.001000	0.8	COMPLETE
# 31	31	0.842510	2021-02-04 14:17:21.178423	2021-02-04 14:18:10.517819	0 days 00:00:49.339396	0.3	0.008	20	250	46	129	0.149526	0.001767	0.8	COMPLETE
# 32	32	0.842561	2021-02-04 14:18:10.520177	2021-02-04 14:19:05.385452	0 days 00:00:54.865275	0.3	0.008	20	210	46	210	0.641484	0.001815	0.8	COMPLETE
# 33	33	0.842461	2021-02-04 14:19:05.387350	2021-02-04 14:19:55.084852	0 days 00:00:49.697502	0.3	0.008	20	283	52	98	0.005502	0.004532	0.4	COMPLETE
# 34	34	0.849797	2021-02-04 14:19:55.087231	2021-02-04 14:22:06.922096	0 days 00:02:11.834865	0.7	0.008	20	283	33	2	0.007228	0.003649	0.4	COMPLETE
# 35	35	0.844002	2021-02-04 14:22:06.925332	2021-02-04 14:23:06.537826	0 days 00:00:59.612494	0.8	0.008	20	267	55	73	0.003325	0.006203	0.4	COMPLETE
# 36	36	0.842594	2021-02-04 14:23:06.539879	2021-02-04 14:23:32.533107	0 days 00:00:25.993228	0.3	0.017	20	299	93	64	0.001055	0.008904	0.4	COMPLETE
# 37	37	0.842638	2021-02-04 14:23:32.537253	2021-02-04 14:24:30.351685	0 days 00:00:57.814432	0.3	0.008	100	221	61	324	0.012162	0.003170	1.0	COMPLETE
# 38	38	0.845129	2021-02-04 14:24:30.353624	2021-02-04 14:24:57.958046	0 days 00:00:27.604422	0.9	0.017	20	181	26	200	0.002179	0.016342	0.6	COMPLETE
# 39	39	0.843104	2021-02-04 14:24:57.960299	2021-02-04 14:25:49.976143	0 days 00:00:52.015844	0.4	0.010	10	193	53	435	0.004191	0.043752	0.4	COMPLETE
# 40	40	0.843218	2021-02-04 14:25:49.978062	2021-02-04 14:26:17.309380	0 days 00:00:27.331318	0.3	0.020	10	71	39	726	0.014720	0.001041	0.5	COMPLETE
# 41	41	0.842554	2021-02-04 14:26:17.311454	2021-02-04 14:27:10.206429	0 days 00:00:52.894975	0.3	0.008	20	244	49	79	1.886850	0.001553	0.8	COMPLETE
# 42	42	0.842386	2021-02-04 14:27:10.208376	2021-02-04 14:28:05.258114	0 days 00:00:55.049738	0.3	0.008	20	285	39	111	6.147695	0.002458	0.8	COMPLETE
# 43	43	0.843595	2021-02-04 14:28:05.260060	2021-02-04 14:29:16.651601	0 days 00:01:11.391541	0.8	0.008	20	287	40	25	6.479868	0.004790	1.0	COMPLETE
# 44	44	0.843388	2021-02-04 14:29:16.653848	2021-02-04 14:30:17.431860	0 days 00:01:00.778012	0.5	0.008	20	266	20	220	6.057562	0.002261	0.6	COMPLETE
# 45	45	0.842482	2021-02-04 14:30:17.433920	2021-02-04 14:31:31.144614	0 days 00:01:13.710694	0.3	0.006	20	300	8	105	1.409994	0.003811	0.8	COMPLETE
# 46	46	0.844395	2021-02-04 14:31:31.146872	2021-02-04 14:32:50.880540	0 days 00:01:19.733668	0.9	0.006	20	156	7	104	3.752531	0.211613	0.4	COMPLETE
# 47	47	0.843564	2021-02-04 14:32:50.883161	2021-02-04 14:34:22.238404	0 days 00:01:31.355243	0.7	0.006	20	285	12	29	0.001660	0.007354	0.8	COMPLETE
# 48	48	0.842449	2021-02-04 14:34:22.240417	2021-02-04 14:35:36.513428	0 days 00:01:14.273011	0.3	0.006	20	263	27	175	1.555404	0.004935	0.7	COMPLETE
# 49	49	0.842411	2021-02-04 14:35:36.515519	2021-02-04 14:37:02.829705	0 days 00:01:26.314186	0.3	0.006	20	264	27	180	0.605466	0.004917	0.7	COMPLETE
# Let's do some Quick Visualization for Hyperparameter Optimization Analysis
# Optuna provides various visualization features in optuna.visualization to analyze optimization results visually
# #plot_optimization_histor: shows the scores from all trials as well as the best score so far at each point.
# optuna.visualization.plot_optimization_history(study)
# #plot_parallel_coordinate: interactively visualizes the hyperparameters and scores
# optuna.visualization.plot_parallel_coordinate(study)
# '''plot_slice: shows the evolution of the search. You can see where in the hyperparameter space your search
# went and which parts of the space were explored more.'''
# optuna.visualization.plot_slice(study)
# #plot_contour: plots parameter interactions on an interactive chart. You can choose which hyperparameters you would like to explore.
# optuna.visualization.plot_contour(study, params=['num_leaves',
#                             'max_depth',
#                             'subsample',
#                             'learning_rate',
#                             'subsample'])
# #Visualize parameter importances.
# optuna.visualization.plot_param_importances(study)
# #Visualize empirical distribution function
# optuna.visualization.plot_edf(study)
# Let's create an LGBMRegressor model with the best hyperparameters
# params=study.best_params
# params['random_state'] = 48
# params['n_estimators'] = 20000
# params['metric'] = 'rmse'
# # I changed min_data_per_groups to cat_smooth beacuse when I used LGBM params in optuna I named cat_smooth
# # as min_data_per_groups (there is no parameter named min_data_per_groups in LGBM !!!)
# params['cat_smooth'] = params.pop('min_data_per_groups')
# params
# {'reg_alpha': 6.147694913504962,
#  'reg_lambda': 0.002457826062076097,
#  'colsample_bytree': 0.3,
#  'subsample': 0.8,
#  'learning_rate': 0.008,
#  'max_depth': 20,
#  'num_leaves': 111,
#  'min_child_samples': 285,
#  'random_state': 48,
#  'n_estimators': 20000,
#  'metric': 'rmse',
#  'cat_smooth': 39}
# columns = categorical_cols+continous_cols
# preds = np.zeros(test.shape[0])
# kf = KFold(n_splits=5,random_state=48,shuffle=True)
# rmse=[]  # list contains rmse for each fold
# n=0
# for trn_idx, test_idx in kf.split(train[columns],train['target']):
#     X_tr,X_val=train[columns].iloc[trn_idx],train[columns].iloc[test_idx]
#     y_tr,y_val=train['target'].iloc[trn_idx],train['target'].iloc[test_idx]
#     model = LGBMRegressor(**params)
#     model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],early_stopping_rounds=100,verbose=False)
#     preds+=model.predict(test[columns])/kf.n_splits
#     rmse.append(mean_squared_error(y_val, model.predict(X_val), squared=False))
#     print(n+1,rmse[n])
#     n+=1
# 1 0.8422516588731099
# 2 0.8454641176530182
# 3 0.8435247420802582
# 4 0.8396584022816828
# 5 0.8409088145101996
# np.mean(rmse)
# 0.8423615470796537
# from optuna.integration import lightgbm as lgb
# lgb.plot_importance(model, max_num_features=10, figsize=(10,10))
# plt.show()
#
# Submission
# sub['target']=preds
# sub.to_csv('submission.csv', index=False)
# # Please If you find this kernel helpful, upvote it to help others see it 😊 😋