"""Fixed-capacity tree candidates; the same recipe also trains every control."""
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb
from catboost import CatBoostRegressor

from run_simple_concat_yield import MemoryGuard
from run_probabilistic_yield_heads import memory_guard

ENGINES=('hgb_base','lgb_base','hgb_smooth','lgb_smooth','xgb_smooth','catboost')


def build(engine,seed,smoke=False):
    iterations=15 if smoke else 1200
    if engine.startswith('hgb'):
        smooth=engine=='hgb_smooth'
        return HistGradientBoostingRegressor(max_iter=15 if smoke else 300 if smooth else 200,
            learning_rate=.03 if smooth else .05,max_leaf_nodes=15 if smooth else 31,
            min_samples_leaf=200 if smooth else 50,l2_regularization=10. if smooth else 1.,
            early_stopping=False,random_state=seed)
    if engine.startswith('lgb'):
        smooth=engine=='lgb_smooth'
        return lgb.LGBMRegressor(n_estimators=iterations,learning_rate=.03,num_leaves=15 if smooth else 31,
            min_child_samples=200 if smooth else 40,subsample=.9,subsample_freq=1,colsample_bytree=.9,
            reg_lambda=10. if smooth else 1.,random_state=seed,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
    if engine=='xgb_smooth':
        return xgb.XGBRegressor(n_estimators=iterations,learning_rate=.03,max_depth=3,min_child_weight=30.,
            subsample=.9,colsample_bytree=.9,reg_lambda=10.,tree_method='hist',device='cuda',
            early_stopping_rounds=60,eval_metric='rmse',random_state=seed,n_jobs=4,callbacks=[MemoryGuard()])
    if engine=='catboost':
        return CatBoostRegressor(iterations=iterations,learning_rate=.03,depth=6,l2_leaf_reg=10.,
            loss_function='RMSE',random_seed=seed,task_type='GPU',devices='0',gpu_ram_part=.08,
            thread_count=4,verbose=False,allow_writing_files=False,early_stopping_rounds=60)
    raise ValueError(engine)


def fit(model,engine,x,y):
    if engine.startswith('lgb'):
        model.fit(x['train'],y['train'],eval_set=[(x['validation'],y['validation'])],
                  callbacks=[lgb.early_stopping(60,verbose=False),lgb.log_evaluation(0)])
    elif engine in ('xgb_smooth','catboost'):
        memory_guard()
        ev=[(x['validation'],y['validation'])] if engine=='xgb_smooth' else (x['validation'],y['validation'])
        model.fit(x['train'],y['train'],eval_set=ev,verbose=False)
        memory_guard()
    else:model.fit(x['train'],y['train'])
