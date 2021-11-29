import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, mean_absolute_percentage_error, roc_curve
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import statsmodels.api as sm

import util.explore as explore_util

def load_embeds(pickle_path, prefix):
  with open(f"../data/transformed_data/{pickle_path}", "rb") as fin:
    embeddings = pickle.load(fin)

  em_ext = {
    "project_id": embeddings["project_ids"],
    f"{prefix}_embed_x_tsne": embeddings[f"tsne"][:, 0],
    f"{prefix}_embed_y_tsne": embeddings[f"tsne"][:, 1],
    f"{prefix}_embed_x_umap": embeddings[f"umap_from_full"][:, 0],
    f"{prefix}_embed_y_umap": embeddings[f"umap_from_full"][:, 1],
    f"{prefix}_embed_x_umap2": embeddings[f"umap_from_pca"][:, 0],
    f"{prefix}_embed_y_umap2": embeddings[f"umap_from_pca"][:, 1]
  }

  embed_df = pd.DataFrame(em_ext)
  return embed_df

def aggregate_proj_fin_type(projectfinancialtype):
    if type(projectfinancialtype) != str:
        return "UNKNOWN"
    
    ptype = projectfinancialtype.upper()
    if "IDA" in ptype and "IBRD" in ptype:
        return "BLEND"
    elif "GRANTS" in ptype and "IBRD" in ptype:
        return "BLEND"
    elif "IDA" in ptype or "GRANTS" in ptype:
        return "IDA"
    elif "IBRD" in ptype:
        return "IBRD"
    else:
        return "OTHER"

def load_projects_with_embeddings(
  ipf_feature_cols=[],
  during_project_features=[],
  reassemble_proj_country_df=False
):

  df = pd.read_json("../data/aggregated_proj.json", orient="index")
  country_panel = pd.read_csv('../data/countrypanel.csv')

  if reassemble_proj_country_df:
    df['boardapprovaldate'] = pd.to_datetime(df['boardapprovaldate'])
    df['closingdate'] = pd.to_datetime(df['closingdate'])
    df['closingyear'] = df.closingdate.dt.year


    new_country_df = pd.read_csv('../data/tmp/project_countries.csv')

    proj_df = df.merge(new_country_df[['project_country', 'panel_country']], 
                  left_on='countryname', right_on='project_country', how='left')

    proj_df = proj_df.drop(columns=['countryname'])
    proj_df = proj_df[proj_df.panel_country.notna()]
    
    practice_count = pd.read_csv("../data/transformed_data/WB_project_practices.zip", compression='zip')
    proj_df = proj_df.merge(practice_count[['proj_id', 'practice_type_code', 'gp_percentage', 'n_practices']], left_on='id', right_on='proj_id', how='left')
    proj_df = proj_df.drop(columns=['proj_id'])
    
    save_transformed_df = True
    if save_transformed_df:
        proj_df.to_csv('../data/transformed_data/projects_with_ccodes.csv')
  else:
    proj_df = pd.read_csv('../data/transformed_data/projects_with_ccodes.csv', index_col=0, low_memory=False)
    proj_df['boardapprovaldate'] = pd.to_datetime(proj_df['boardapprovaldate'])
    proj_df['closingdate'] = pd.to_datetime(proj_df['closingdate'])

  aux_proj_data = pd.read_csv('../data/transformed_data/aux_project_data.zip', compression='zip')

  pdo_embed_df = load_embeds("title_pdo_embeds_reduced.pkl", "pdo")
  dli_embed_df = load_embeds("dli_embeddings_reduced.pkl", "dli")
  embed_cols = [col for col in list(pdo_embed_df.columns) + list(dli_embed_df.columns) if col != "project_id"]

  sector_df = pd.read_csv('../data/transformed_data/WB_project_sectors.zip', compression='zip').rename(columns={ 'proj_id': 'id' })
  main_sector_df = sector_df[sector_df.flag_main_sector == 1]

  sector_df['sq_percent'] = sector_df['sector_percentage'] ** 2
  hhi_df = sector_df.groupby('id', as_index=False).agg(hhi=('sq_percent', 'sum'))
  hhi_df['hhi_clipped'] = hhi_df['hhi'].clip(upper=(100 * 100))

  def assemble_df(proj_feature_cols):
    ndf = proj_df.merge(aux_proj_data[['projid'] + proj_feature_cols], left_on='id', right_on='projid', how='left')

    ndf = ndf.merge(pdo_embed_df, left_on="id", right_on="project_id", how="left")
    ndf = ndf.merge(dli_embed_df, left_on="id", right_on="project_id", how="left")
    ndf[embed_cols] = ndf[embed_cols].fillna(0)

    ndf = ndf.merge(main_sector_df[['id', 'sector_code', 'sector_percentage', 'parent_sector_name']], how='left')
    ndf = ndf.merge(hhi_df, how='left')

    ndf["financing_type"] = ndf.projectfinancialtype.apply(aggregate_proj_fin_type)

    ndf["financing_instr"] = ndf.lendinginstr.replace({
        "Sector Investment and Maintenance Loan": "Specific Investment Loan",
        0: "UNIDENTIFIED"
    })

    narrow_sector_features = ['sector1', 'sector2', 'sector3', 'sector4', 'sector5']

    sector_count_df = ndf[['id'] + narrow_sector_features]
    sector_count_df[narrow_sector_features] = sector_count_df[narrow_sector_features].notna()
    sector_count_df['number_sectors'] = sector_count_df[narrow_sector_features].sum(axis=1)

    ndf = ndf.merge(sector_count_df[['id', 'number_sectors']])

    ndf["focused_project"] = (ndf.sector_percentage > 90) | (ndf.number_sectors == 1)
    ndf["scattered_project"] = (~ndf.focused_project) & ((ndf.sector_percentage < 60) | (ndf.number_sectors > 2))

    return ndf

  approv_df = assemble_df(ipf_feature_cols)
  review_df = assemble_df(during_project_features)

  return country_panel, approv_df, review_df

def assemble_input_df(ndf, relevant_feature_cols, country_panel, 
                        sector_features=['sector1', 'sector2', 'sector3', 'sector4', 'sector5', 'theme1', 'theme2']):
    wdf = ndf[relevant_feature_cols].fillna(0)
    wdf['all_sectors_theme_words'] = wdf[sector_features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).str.lower()
    wdf['is_health_project'] = wdf.all_sectors_theme_words.str.contains('health')
    wdf['is_education_project'] = wdf.all_sectors_theme_words.str.contains('edu')

    data = wdf.merge(country_panel.drop(columns=['regionname']), 
                                    left_on=['panel_country', 'closingyear'], right_on=['countryname', 'year'])
    data = data.drop(columns=['countryname', 'year'])
    data = data[data.closingyear.notna()]
    data['pdo_length'] = data['pdo'].str.len().fillna(0)

    data = data.rename(columns = { 'project_country': 'country', 'closingyear': 'year' })

    return data

# slightly redundant to do each time, but different features may have different missingness
def construct_residual_df(data, project_feature_cols):
  health_target = 'mortality_under5_lag-5'

  health_to_lag = {
      'mortality_under5': -5,
      'hiv_prevalence': -5,
      'conflict': -5
  }

  health_observed_X_cols = [
      'gdp_pc_ppp',
      'fertility',
      'population',
      'physicians_rate',
      'female_adult_literacy',
      'access_water',
      'access_sanitation',
      'hiv_prevalence_lag-5'
  ]

  edu_target = 'edu_aner_lag-5'

  edu_to_lag = {
      'edu_aner': -5,
      'edu_share_gov_exp': -5,
      'edu_pupil_teacher': -5,
      'young_population': -5,
      'cash_surplus_deficit': -5, 
      'inflation': -5,
      'trade_share_gdp': -5,
      'freedom_house': -5
  }

  edu_observed_X_cols = [f"{obs_col}_lag-5" for obs_col in edu_to_lag.keys() if obs_col != "edu_aner"]

  health_results = end_to_end_project_eval(
    data, "health", "mortality_under5_lag-5", health_to_lag,
    observed_X_cols=health_observed_X_cols, 
    loan_feature_cols=project_feature_cols,
    inverted_outcome=True
  )

  edu_results = end_to_end_project_eval(
    data, "edu", "edu_aner_lag-5", edu_to_lag,
    observed_X_cols=edu_observed_X_cols, 
    loan_feature_cols=project_feature_cols,
    inverted_outcome=True
  )

  consolidated_df = pd.concat((health_results["residual_df"], edu_results["residual_df"]))
  consolidated_df = consolidated_df.fillna(0)

  return consolidated_df, health_results, edu_results

def sum_feature_imp(category_name, feature_imp, exp_features):
    return sum([feature_imp[col] for col in exp_features if col.startswith(category_name)])

def extract_feature_imp(est, orig_features, exp_features):
    feature_imp = { col: est.feature_importances_[i] for i, col in enumerate(exp_features) }
    summed_feature_imp = { col: sum_feature_imp(col, feature_imp, exp_features) for col in orig_features }
    return feature_imp, summed_feature_imp

def fit_score_model(X, y, est, classification=False, random_seed=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y if classification else None, random_state=random_seed)
    print("Size of X train: ", len(X_train), " and X test: ", len(X_test), " and width: ", len(X_train.columns))
    
    est.fit(X_train, y_train)
    scores = { 'default_score': est.score(X_test, y_test) }
    if classification:
        true_pred = est.predict_proba(X_test)[:, 1]
        scores['fscore_etc'] = precision_recall_fscore_support(y_test, est.predict(X_test), average="binary")
        scores['roc_auc'] = roc_auc_score(y_test, true_pred)
        scores['roc_curve'] = roc_curve(y_test, true_pred)
        scores['roc_auc_train'] = roc_auc_score(y_train, est.predict_proba(X_train)[:, 1])
    else:
        scores['mape'] = mean_absolute_percentage_error(y_test, est.predict(X_test))
        scores['mape_train'] = mean_absolute_percentage_error(y_train, est.predict(X_train))
        scores['r2_train'] = est.score(X_train, y_train)

    test_data = { "X_test": X_test, "y_test": y_test }
    return est, scores

def end_to_end_project_eval(all_data, sector_key_word, target_col, variables_to_lag, observed_X_cols, loan_feature_cols, 
                            regressor=RandomForestRegressor, classifier=RandomForestClassifier, inverted_outcome=False):
    sector_data = all_data.copy()
    
    for var in variables_to_lag:
        sector_data = explore_util.lag_variable_simple(sector_data, var, variables_to_lag[var])
    
    sector_data['is_sector_project'] = sector_data.all_sectors_theme_words.str.contains(sector_key_word)
    sector_data = sector_data[sector_data.is_sector_project]
    print("Sector projects data: ", len(sector_data), " versus all projects: ", len(all_data))
    
    sdata = sector_data[['id'] + observed_X_cols + [target_col]]
    sdata = sdata.dropna()
    print("Clean observations: ", len(sdata))

    # print("Pre scaling: ", sdata[observed_X_cols[:2]].describe())
    observation_scaler = StandardScaler()
    sdata[observed_X_cols] = observation_scaler.fit_transform(sdata[observed_X_cols])
    # print("Shape of endog: ", sdata[target_col].shape, " and exog: ", sm.add_constant(sdata[observed_X_cols]).shape)
    res_est = sm.OLS(endog=sdata[target_col], exog=sm.add_constant(sdata[observed_X_cols])).fit()
    print("Naive R squared of partialling out phase: ", res_est.rsquared, " and f_p: ", res_est.f_pvalue)
    # print("Post scaling: ", sdata[observed_X_cols[:2]].describe())
    
    target_resid = f"residual_target"
    sdata[target_resid] = res_est.resid
    
    forest_data = sdata[['id', target_resid]].merge(all_data[['id'] + loan_feature_cols], how='inner')
#     print(forest_data.isna().sum())
    pre_scale_target_desc = forest_data[target_resid].describe()
    # print("Descriptive stats for target: ", pre_scale_target_desc)
    
    numeric_cols = forest_data.select_dtypes(include=np.number).columns.tolist()
    treatment_scaler = StandardScaler()
    forest_data[numeric_cols] = treatment_scaler.fit_transform(forest_data[numeric_cols])

    categorical_cols = [col for col in loan_feature_cols if col not in numeric_cols]
    forest_data = pd.get_dummies(forest_data, columns=categorical_cols)

    forest_data = forest_data.dropna()
    print("Clean within project characteristics: ", len(forest_data))
    
    pos_std_dev_threshold = 0.1
    forest_data[f'{target_resid}_above_threshold'] = (
        forest_data[target_resid] > pos_std_dev_threshold if not inverted_outcome else 
            forest_data[target_resid] < pos_std_dev_threshold
    )
    print("Projects with residual above mean: ", len(forest_data[forest_data[target_resid] > 0]))
    print("Projects with positive residual above threshold: ", len(forest_data[forest_data[target_resid] > pos_std_dev_threshold]))
    
    nreg = regressor()
    nest = classifier()
    
    X = forest_data.drop(columns=['id', target_resid, f'{target_resid}_above_threshold'])
    
    y_reg = forest_data[target_resid]
    y_class = forest_data[f'{target_resid}_above_threshold']
    
    reg_fit, reg_scores = fit_score_model(X, y_reg, nreg)
    bin_est, bin_scores = fit_score_model(X, y_class, nest, classification=True)
    
    all_col_imp, summed_imp = extract_feature_imp(bin_est, loan_feature_cols, X.columns)
    summed_imp = sort_imp(summed_imp)
    
    return {
        "partial_out_model": res_est,
        "residual_regressor": reg_fit,
        "residual_classifier": bin_est,
        "regression_scores": reg_scores,
        "classifier_scores": bin_scores,
        "pre_scale_target_stats": pre_scale_target_desc,
        "summed_importances": summed_imp,
        "all_importances": all_col_imp,
        "residual_df": forest_data
    }

def sort_imp(summed_imp):
  return { feature: score for feature, score in sorted(summed_imp.items(), key=lambda item: item[1], reverse=False )}

def drop_agg_cols(X, columns_to_drop):
    split_cols_to_drop = []
    for agg_col in columns_to_drop:
        split_cols_to_drop += [col for col in X.columns if agg_col in col]
    return X.drop(columns=split_cols_to_drop)

def run_residual_reg(consolidated_df, probe_feature_cols, columns_to_drop=[], reg=RandomForestRegressor(), clf=RandomForestClassifier(), random_seed=None):
    X = consolidated_df.drop(columns=['id', "residual_target", f'residual_target_above_threshold'])
    if len(columns_to_drop) > 0:
        split_cols_to_drop = []
        for agg_col in columns_to_drop:
            split_cols_to_drop += [col for col in X.columns if agg_col in col]
        X = X.drop(columns=split_cols_to_drop)
        
    y_reg = consolidated_df["residual_target"]
    y_class = consolidated_df['residual_target_above_threshold']

    reg_fit, reg_scores = fit_score_model(X, y_reg, reg, random_seed=random_seed)
    bin_est, bin_scores = fit_score_model(X, y_class, clf, classification=True, random_seed=random_seed)

    all_col_imp, summed_imp = extract_feature_imp(bin_est, probe_feature_cols, X.columns)
    summed_imp = { feature: score for feature, score in sorted(summed_imp.items(), key=lambda item: item[1], reverse=True)}

    models = [reg_fit, bin_est]

    return bin_scores, reg_scores, summed_imp, models

def conduct_drop_one(data=None, clf=None, all_feature_cols=None, feature_imp=None, 
    cols_to_ablate=None, cols_to_exclude=None, ref_bin_scores=None, ref_reg_scores=None, random_seed=None):
  
  by_col_ablation = []
  print("Initiating drop one tests, for columns: ", cols_to_ablate)

  for col in cols_to_ablate:
    print(".", end="")
    cols_to_drop = [col] + cols_to_exclude if type(col) == str else col + cols_to_exclude
    ablation_results = run_residual_reg(data, all_feature_cols, columns_to_drop=cols_to_drop, clf=clf, random_seed=random_seed)
    
    roc_score = ablation_results[0]["roc_auc"]
    penalty_score = ref_bin_scores["roc_auc"] - roc_score
    r2_score = ablation_results[1]["r2_train"]
    penalty_r2 = ref_reg_scores["r2_train"] - r2_score
    col_name = col if type(col) == str else col[0][:(col[0].find("x") - 1)]
    by_col_ablation.append({ "col": col_name, "roc_score": roc_score, "penalty_score": penalty_score, "r2_score": r2_score, "penalty_r2":  penalty_r2})
  
  print(" done") # for new line
  result_df = pd.DataFrame(by_col_ablation).sort_values(by=["penalty_score", "penalty_r2"])
  result_df["feat_imp"] = result_df.col.apply(lambda col: feature_imp.get(col, 0))
  result_df.sort_values(by=["penalty_r2"], ascending=False)
  
  return result_df