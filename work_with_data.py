import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score

PATH_TO_TRAIN_CSV = "D:/CI/CI_labs/lab_1/train_energy_data.csv"
PATH_TO_TEST_CSV = "D:/CI/CI_labs/lab_1/test_energy_data.csv"
INDENT = '\n\n\t\t\t'

# helpers
def plotify(results_df):
    subset = results_df.iloc[:20]
    x_axis = range(len(subset))
    
    plt.figure(figsize=(12, 6))
    
    plt.scatter(x_axis, subset['Real Value'], color='red', label='Real Fact', zorder=5)
    
    plt.plot(x_axis, subset['Predicted'], color='blue', label='Prediction', linestyle='--')
    
    plt.fill_between(x_axis, 
                     subset['Lower'], 
                     subset['Upper'], 
                     color='blue', alpha=0.2, label='95% Confidence Interval')
    
    plt.title('Probabilistic Regression Forecast (Bayesian Ridge)')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Energy Consumption')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
def show_metrics(test, pred, name = "", show=True):
    mae = mean_absolute_error(test, pred)
    rmse = np.sqrt(mean_squared_error(test, pred))
    r2 = r2_score(test, pred)
    if show:
        print(f"{INDENT} {name} Metrics:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2: {r2:.2f}")
    return mae, rmse, r2

def data_report(raw_data):
    print(f"{INDENT} Example of data:")
    print(raw_data.head())

    print(f"{INDENT} Core information about data:")
    raw_data.info()
    print(f"{INDENT} Analysis main info about data:")
    print(raw_data.describe())

    print(f"{INDENT} Checking if data have NULL values:")
    print(raw_data.isnull().sum())

def data_cleaning(raw_data):
    mapping_days = {
        "Weekend": 0,
        "Weekday": 1
    }
    raw_data['Day of Week'] = raw_data['Day of Week'].map(mapping_days)

    raw_data = pd.get_dummies(raw_data, columns=['Building Type'], drop_first=True, dtype=int) 

    target_col = 'Energy Consumption'

    x_values = raw_data.drop(columns=target_col)
    y_values = raw_data[target_col]
    
    return x_values, y_values

def preprocesing(show=False):
    # Import data
    train_raw = pd.read_csv(PATH_TO_TRAIN_CSV)
    test_raw = pd.read_csv(PATH_TO_TEST_CSV)

    if show:
       data_report(train_raw)
    
    # data scaling
    x_train, y_train = data_cleaning(train_raw)
    x_test, y_test = data_cleaning(test_raw)

    scaler = StandardScaler()
    num_colums = ['Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature']

    x_train[num_colums] = scaler.fit_transform(x_train[num_colums] )
    x_test[num_colums] = scaler.transform(x_test[num_colums])

    if show:
        print(f"{INDENT} Checking results:")
        print("Shape of Train:", x_train.shape)
        print("Shape of Test:", x_test.shape)
    
    return x_train, x_test, y_train, y_test

def basic_regression(x_train, x_test, y_train, y_test, show=True, is_need_plot=True):
    # Linear Regression
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

def model_execution(model, x, y, x_t, y_t, show = True):
    model.fit(x, y)
    pred = model.predict(x_t)
    title =  str(type(model).__name__) + " Method" 
    
    if show:
        show_metrics(pred, y_t, title)
    train_pred = model.predict(x)
    residuals = y - train_pred
    std_dev = np.std(residuals) 
    
    lower = pred - 1.96 * std_dev
    upper = pred + 1.96 * std_dev
    
    df_results = pd.DataFrame({
        'Model': title,
        'Real Value': np.array(y_t),
        'Predicted': pred,
        'Lower Bound': lower,
        'Upper Bound': upper
    })
    
    return df_results
    
def ensemble_models(x_train, x_test, y_train, y_test):
    estimators = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('gb', GradientBoostingRegressor(random_state=42)),
        ('ridge', Ridge())
    ]
    
    models = [
        Ridge(),
        RandomForestRegressor(random_state=42),
        GradientBoostingRegressor(random_state=42),
        StackingRegressor(estimators=estimators, final_estimator=Ridge())
    ]
    
    results = []
    for model in models:
        data_frame = model_execution(model, x_train, y_train, x_test, y_test)
        results.append(data_frame)
            
    print(f"{INDENT} Models comparation")
    
    compare_df = pd.concat(results, ignore_index=True)
    print(compare_df.groupby('Model').head(1).to_string())

def to_clasification(y, y_to_test, discretization=3):  
    y_train, bin_edges = pd.cut(
        x=y,
        bins=discretization,
        labels=[i for i in range(discretization)],
        retbins=True
    )
    y_test = pd.cut(
        x=y_to_test,
        bins=bin_edges,
        labels=[i for i in range(discretization)],
        include_lowest=True
    )
    print(y_train.value_counts())
    return y_train, y_test    
    
def print_class_metrics(name:str, y, y_pred):
    print(f"{INDENT} {name} Метрики:")
    report = classification_report(y, y_pred)
    print(report)
    
def classification(x_train, x_test, y_train, y_test, discretization, show):
    y_train, y_test = to_clasification(y_train, y_test, discretization)
    
    log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    gbc = GradientBoostingClassifier(random_state=42)
    
    models = [log_reg, rf, gbc]
    results = []
    
    for model in models:
        model.fit(x_train, y_train)
        
        prediction = model.predict(x_test)
        results.append(prediction)
    if show:
        names = [type(m).__name__ for m in models ]   
        for name, y_pred in zip(names, results):
            print_class_metrics(name, y_test, y_pred)
    
def hist_model(confidence, model):
    plt.figure(figsize=(8, 5))
    plt.hist(confidence, bins=20, color='skyblue', edgecolor='black')

    plt.title(f"Гістограма впевненості моделі: {type(model).__name__}")
    plt.xlabel("Ймовірність (Впевненість у своєму виборі)")
    plt.ylabel("Кількість будівель")
    plt.axvline(x=0.33, color='red', linestyle='--', label='Повна невпевненість (вгадування)')
    plt.legend()
    plt.show()
    
def pred_res(x, x_t, y, y_t, show=True, show_plot=True):
    y, y_t = to_clasification(y, y_t)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(x, y)
    probabilities = model.predict_proba(x_t)
    confidence = np.max(probabilities, axis=1)
    
    underfit_model = DecisionTreeClassifier(max_depth=1, random_state=42)
    underfit_model.fit(x, y)
    under_pred = underfit_model.predict_proba(x_t)
    under_confidence = np.max(under_pred, axis=1)
    
    overfit_model = DecisionTreeClassifier(max_depth=None, random_state=42)
    overfit_model.fit(x, y)
    over_pred = overfit_model.predict_proba(x_t)
    over_confidence = np.max(over_pred, axis=1)
    if show_plot:
        hist_model(under_confidence , underfit_model)
        hist_model(confidence, model)
        hist_model(over_confidence, overfit_model)
    
def best_model_tunning(x_train, x_test, y_train, y_test):
    param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
    }
    
    y, y_t = to_clasification(y_train, y_test)   
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    rf.fit(x_train, y)
    pred = rf.predict(x_test)
    print_class_metrics("Default RF", y_t, pred)
    
    print(f"{INDENT} START OPTIMIZATION ATTEMPT")
    grid_search.fit(x_train, y)
    print(f"BEST PARAMS: {grid_search.best_estimator_}")
    
    best_rf_model = grid_search.best_estimator_
    optimized_pred = best_rf_model.predict(x_test)
    print("\n=== Метрики ОПТИМІЗОВАНОЇ моделі RandomForest ===")
    print_class_metrics("Optimized RF", y_t, optimized_pred)

def metrics_conversion(x_train, x_test, y_train, y_test):
    ridge = Ridge()
    ridge.fit(x_train, y_train)
    pred_ridge = ridge.predict(x_test)

    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')

    y_train_2d = y_train.values.reshape(-1, 1)
    discretizer.fit(y_train_2d)

    y_test_2d = y_test.values.reshape(-1, 1)
    y_test_kbins = discretizer.transform(y_test_2d)

    pred_ridge_2d = pred_ridge.reshape(-1, 1)
    pred_kbins = discretizer.transform(pred_ridge_2d)

    reg_to_class_acc = accuracy_score(y_test_kbins, pred_kbins)
    print(f"Точність регресійної моделі (Ridge) після перетворення у класи: {reg_to_class_acc:.4f} ({reg_to_class_acc * 100}%)")
    
    train_classes = discretizer.transform(y_train_2d).flatten()
    temp_df = pd.DataFrame({'Real_Y': y_train.values, 'Class': train_classes})
    
    class_means = temp_df.groupby('Class')['Real_Y'].mean()
    print(f"\nСереднє споживання для кожного класу (кВт):")
    for cls, mean_val in class_means.items():
        print(f"Клас {int(cls)}: {mean_val:.2f}")

    pred_means = [class_means[int(c)] for c in pred_kbins.flatten()]

    discretized_mae = mean_absolute_error(y_test, pred_means)

    print(f"\nОригінальне MAE моделі Ridge: 1.22")
    print(f"Оціночне MAE через дискретизовані прогнози: {discretized_mae:.2f}")
    
def main():
    # Task 1
    x_train, x_test, y_train, y_test = preprocesing(show=False)

    # Task 2
    basic_regression(x_train, x_test, y_train, y_test, is_need_plot=False, show=True)
    
    # Task 3
    ensemble_models(x_train, x_test, y_train, y_test)
    
    # Task 4
    classification(x_train, x_test, y_train, y_test, 3, True)
    classification(x_train, x_test, y_train, y_test, 4, True)
    
    # Task 5
    pred_res(x_train, x_test, y_train, y_test, show_plot=False)
    
    # Task 6
    best_model_tunning(x_train, x_test, y_train, y_test)
    
    # Task 7
    metrics_conversion(x_train, x_test, y_train, y_test )
    
if __name__ == "__main__":
    main()