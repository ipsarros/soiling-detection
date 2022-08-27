import numpy as np
import pandas as pd
import modules.learning as le
import modules.statistics as st
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import bernoulli, binom
from sklearn import metrics


def get_line_and_slope2(values):
    """
    Fits a line on the 2-dimensional graph of a regular time series, defined by a sequence of real values. 

    Args:
        values: A list of real values.

    Returns: 
        line: The list of values as predicted by the linear model.
        slope: Slope of the line.
        intercept: Intercept of the line.   
    """

    ols = TheilSenRegressor(random_state=0)
    X = np.arange(len(values)).reshape(-1,1)
    y = values.reshape(-1,1)
    
    ols.fit(X, y.ravel())
    line = ols.predict(X)
    slope = ols.coef_.item()
    intercept = ols.intercept_.item()
    return line, slope, intercept


def errors_at_rains(df, dates_rain_start, dates_rain_stop, model, target, feats, w1, w2):
    errors_br = np.empty((dates_rain_start.size, 6))
    errors_ar = np.empty((dates_rain_start.size, 6))
    for i in range(dates_rain_start.size):
        d1 = dates_rain_start.iloc[i]
        d0 = d1 - pd.Timedelta(days=w1)
        d2 = dates_rain_stop.iloc[i]
        d3 = d2 + pd.Timedelta(days=w2)
        df_ar = df[d2:d3]
        df_br = df[d0:d1]
        try:
            y_pred_ar = le.predict(df_ar, model, feats, target)
            y_pred_br = le.predict(df_br, model, feats, target)
            errors_ar[i,:] = st.score(df_ar[target].array, y_pred_ar)
            errors_br[i,:] = st.score(df_br[target].array, y_pred_br)
        except:
            errors_ar[i,:] = [np.nan]*6
            errors_br[i,:] = [np.nan]*6
            print(f'Rain {i} failed. Rain between: {d1} and {d2}')
    return errors_br, errors_ar        

def errors_at_rains2(df, dates_rain_start, dates_rain_stop, target, feats, w1, w2, w3):
    errors_br = np.empty((dates_rain_start.size, 6))
    errors_ar = np.empty((dates_rain_start.size, 6))
    for i in range(dates_rain_start.size):
        d1 = dates_rain_start.iloc[i]
        d2 = dates_rain_stop.iloc[i]
        try:
            y_pred_train, score_train, y_pred_val, errors_br[i,:], y_pred_test, errors_ar[i,:] = le.changepoint_scores(df, feats, target, d1, d2, w1, w2, w3)
        except:
            errors_ar[i,:] = [np.nan]*6
            errors_br[i,:] = [np.nan]*6
            print(f'Rain {i} failed. Rain between: {d1} and {d2}')
    return errors_br, errors_ar  

def output_changepoints(scores, indices, dates_rain_start, dates_rain_stop, errors_br, errors_ar, error_name_br, error_name_ar, precip):
    start_dates = []
    end_dates = []
    all_prec = []
    all_errors_br = []
    all_errors_ar = []
    all_scores = []
    prec1 = []
    prec2 = []
    types = []
    ids = []
    #for r, i in enumerate(indices[j]):
    for i in indices:
        d1 = dates_rain_start.iloc[i]
        d2 = dates_rain_stop.iloc[i]
        all_errors_br.append(errors_br[i])
        all_errors_ar.append(errors_ar[i])
        precip2 = precip.loc[d1:d2][1:-1].values
        start_dates.append(d1)
        end_dates.append(d2)
        prec1.append(precip2.max())
        prec2.append(precip2.mean())
        ids.append(i)    
        all_scores.append(scores[i])
        #(MAPE before rain/MAPE after rain)
    #df_events_output.append(pd.DataFrame.from_dict({"Αrray": j+1,"Score (Percentage difference of median error)": all_scores, "Rain id": ids, "Starting date": start_dates, "Ending date": end_dates, "Max precipitation": np.round(prec1,2),  "Mean precipitation": np.round(prec2,2), error_names[error_br_column]+" before rain": all_errors_br,  error_names[error_ar_column]+" after rain": all_errors_ar}))    
    return pd.DataFrame.from_dict({"Score": all_scores, "Rain id": ids, "Starting date": start_dates, "Ending date": end_dates, "Max precipitation": np.round(prec1,2),  "Mean precipitation": np.round(prec2,2), error_name_br+" before rain (true-pred)": all_errors_br,  error_name_ar+" after rain (true-pred)": all_errors_ar})    
    
def train_on_reference_points_cluster(df_list, w_train, ref_points, feats, target, random_state=0):
    df_train = pd.DataFrame([])
    df_val = pd.DataFrame([])
    for df in df_list:
        for idx in range(ref_points.size):
            d_train_stop = pd.to_datetime(ref_points[idx]) + pd.Timedelta(days=w_train)
            df_tmp = df.loc[ref_points[idx]:str(d_train_stop)]
            df_tmp2 = df_tmp.sample(frac=1, random_state=random_state) # added random state for reproducibility during experiments
            size_train = int(len(df_tmp2) * 0.80)
            df_train = df_train.append(df_tmp2[:size_train])
            df_val = df_val.append(df_tmp2[size_train:])

    model, y_pred_train, r_sq_train, mae_train, me_train, mape_train, mpe_train, Me_train = fit_linear_model(df_train, feats, target)
    y_pred_val = predict(df_val, model, feats, target)
    r_sq_val, mae_val, me_val, mape_val, mpe_val, Me_val = st.score(df_val[target].values, y_pred_val)
    training_scores = np.array([r_sq_train, mae_train, me_train, mape_train, Me_train])
    validation_scores = np.array([r_sq_val, mae_val, me_val, mape_val, mpe_val, Me_val])

    print('Training Metrics:')
    print(f'MAE:{training_scores[1]:.3f} \nME(true-pred):{training_scores[2]:.3f} \nMAPE:{training_scores[3]:.3f} \nR2: {training_scores[0]:.3f}\n')
    print('Validation Metrics:')
    print(f'MAE:{validation_scores[1]:.3f} \nME(true-pred):{validation_scores[2]:.3f} \nMAPE:{validation_scores[3]:.3f} \nMPE:{validation_scores[4]:.3f} \nR2: {validation_scores[0]:.3f}\n')
    return model, training_scores, validation_scores

def clean_points_sampling(df):
    print(min(df.index))
    for x in df:
        print(x)
    return(df)
def train_on_reference_points_randomized(df, w_train, ref_points, feats, target, random_state=0):
    df_train = pd.DataFrame([])
    df_val = pd.DataFrame([])
    for idx in range(ref_points.size):
        d_train_stop = pd.to_datetime(ref_points[idx]) + pd.Timedelta(days=w_train)
        df_tmp = df.loc[ref_points[idx]:str(d_train_stop)]
        df_tmp2 = df_tmp.sample(frac=1, random_state=random_state) # added random state for reproducibility during experiments
        size_train = int(len(df_tmp2) * 0.80)
        df_train = df_train.append(clean_points_sampling(df_tmp2[:size_train]))
        df_val = df_val.append(df_tmp2[size_train:])

    model, y_pred_train, r_sq_train, mae_train, me_train, mape_train, mpe_train, Me_train = le.fit_linear_model(df_train, feats, target)
    y_pred_val = le.predict(df_val, model, feats, target)
    r_sq_val, mae_val, me_val, mape_val, mpe_val, Me_val = st.score(df_val[target].values, y_pred_val)
    training_scores = np.array([r_sq_train, mae_train, me_train, mape_train, Me_train])
    validation_scores = np.array([r_sq_val, mae_val, me_val, mape_val, mpe_val, Me_val])

    print('Training Metrics:')
    print(f'MAE:{training_scores[1]:.3f} \nME(true-pred):{training_scores[2]:.3f} \nMAPE:{training_scores[3]:.3f} \nR2: {training_scores[0]:.3f}\n')
    print('Validation Metrics:')
    print(f'MAE:{validation_scores[1]:.3f} \nME(true-pred):{validation_scores[2]:.3f} \nMAPE:{validation_scores[3]:.3f} \nMPE:{validation_scores[4]:.3f} \nR2: {validation_scores[0]:.3f}\n')
    return model, training_scores, validation_scores
