from config import INDENT
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge, BayesianRidge

# import numpy as np
# from sklearn.model_selection import GridSearchCV


def model_testing(model, x_train, x_test, y_train, y_test):
    pass


# def show_metrics(test, pred, name = "", show=True):
#     mae = mean_absolute_error(test, pred)
#     rmse = np.sqrt(mean_squared_error(test, pred))
#     r2 = r2_score(test, pred)
#     if show:
#         print(f"{INDENT} {name} Metrics:")
#         print(f"MAE: {mae:.2f}")
#         print(f"RMSE: {rmse:.2f}")
#         print(f"R2: {r2:.2f}")
#     return mae, rmse, r2

def basic_regression(x_train, x_test, y_train, y_test, show=True, is_need_plot=True):   
    models = {
        LinearRegression(),
        Ridge(),
        BayesianRidge()
    }
    
    lr_model = LinearRegression()

    lr_model.fit(x_train, y_train)

    y_pred = lr_model.predict(x_test)

    lr_mae, lr_rmse, lr_r2 = show_metrics(y_test, y_pred, show=False)
    # Ridge regression
    rr_model = Ridge()

    rr_model.fit(x_train, y_train)

    y_pred = rr_model.predict(x_test)

    rr_mae, rr_rmse, rr_r2 = show_metrics(y_test, y_pred, show)

    if show:
        print(f"{INDENT} Comparing LR and RR")

        w_name = 25  
        w_num = 15   

        print("\n|" + "="*63 + "|")
        print(f"| {'Feature':<{w_name}} | {'Linear':>{w_num}} | {'Ridge':>{w_num}} |")
        print("|" + "-"*63 + "|")

        for feat, l_c, r_c in zip(x_train.columns, lr_model.coef_, rr_model.coef_):
            print(f"| {feat:<{w_name}} | {l_c:>{w_num}.4f} | {r_c:>{w_num}.4f} |")
        print(f"| {'MSE':<{w_name}} | {lr_rmse:>{w_num}.4f} | {rr_rmse:>{w_num}.4f} |")
        print(f"| {'R2':<{w_name}} | {lr_r2:>{w_num}.4f} | {rr_r2:>{w_num}.4f} |")
        print("|" + "="*63 + "|")
        
    m_rid = {
        'mae':rr_mae,
        'rmse': rr_rmse,
        'r2' : rr_r2
        }
    
    # 2.2
    bayesian_model = BayesianRidge()
    bayesian_model.fit(x_train, y_train)
    
    y_mean, y_std = bayesian_model.predict(x_test, return_std=True)
    
    lower_bound = y_mean - 1.96 * y_std
    upper_bound = y_mean + 1.96 * y_std
    results_df = pd.DataFrame({
            'Real Value': y_test,
            'Predicted': y_mean,
            'Lower': lower_bound,
            'Upper': upper_bound,
            'Uncertainty': y_std
    })
    if show:
        print(f"{INDENT} Prediction of probability with borders")
        print(results_df.head(10).round(2).to_string())

        in_bounds = (results_df['Real Value'] >= results_df['Lower']) & \
                    (results_df['Real Value'] <= results_df['Upper'])

        print(f"{INDENT} Persantage of scoring interval 95%: {in_bounds.mean() * 100:.2f}%")
    
    if is_need_plot:
        plotify(results_df)
    
    # 2.3
    mae_trst, rmse_trst, rw_trst = show_metrics(y_test, y_mean, show)
    
    # 2.4
    bayes_model = BayesianRidge()
    
    grid_params = {
    'alpha_1': [1e-6, 1e-5, 1e-4],
    'alpha_2': [1e-6, 1e-5, 1e-4], 
    'lambda_1': [1e-6, 1e-5, 1e-4],
    'lambda_2': [1e-6, 1e-5, 1e-4]
    }
   
    search_bayes = GridSearchCV(bayes_model, grid_params, scoring='r2', n_jobs=-1, cv=5, verbose=0)
    
    search_bayes.fit(x_train, y_train)
    if show:
        
        print(f"{INDENT} Best params {search_bayes.best_params_}")
    
    best_bayes_model = search_bayes.best_estimator_
    
    y_pred_opt, y_std_opt = best_bayes_model.predict(x_test, return_std=True)
    
    # 2.5 
    lower_opt =  y_pred_opt - 1.96 *  y_std_opt
    upper_opt = y_pred_opt + 1.96 *  y_std_opt
    if show:
        results_df_tunned = pd.DataFrame({
            'Real Value': y_test,
            'Predicted': y_pred_opt,
            'Lower': lower_opt,
            'Upper': upper_opt,
            'Uncertainty': y_std_opt
        })

        print(results_df_tunned.head(10).round(2).to_string())
    
    mae_opt, mse_opt, r2_opt = show_metrics(y_test, y_pred_opt, show)
    
    comparing = pd.DataFrame({
          'Metric': ['MAE', 'MSE', 'R2'],
          'Base Model(Ridge)': [rr_mae, rr_rmse, rr_r2],
          'Base Bayesian model': [mae_trst, rmse_trst, rw_trst ],
          'Optimized Bayesian': [mae_opt, mse_opt, r2_opt]
    })
    
    print(f"{INDENT} Final results for task 2:")
    print(comparing.round(10).to_string(index=False))