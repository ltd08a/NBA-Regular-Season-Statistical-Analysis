
# Create a function to generate and display a correlation matrix for the full dataset
def gen_corr_matrix(df, threshold=0.8):
    """
    Generate a correlation matrix for the given DataFrame and return a new DataFrame containing 
    features with strong correlations.
    
    Parameters:
    df (pd.DataFrame): The DataFrame for which to compute the correlation matrix.
    threshold (float): The threshold above which correlations are considered strong.
        
    Returns:
    correlation_matrix (pd.DataFrame): The correlation matrix of the data.
    strong_correlations (pd.DataFrame): DataFrame containing features with strong correlations.
    """
    # import the pandas library
    import pandas as pd
    
    # Create a correlation matrix
    correlation_matrix = df.corr()
    
    # Unstack the correlation matrix to create a Series of correlation pairs
    correlation_pairs = correlation_matrix.unstack()
    
    # Convert the Series to a DataFrame and reset the index
    correlation_pairs_df = pd.DataFrame(correlation_pairs, columns=['Correlation']).reset_index()
    correlation_pairs_df.columns = ['Feature1', 'Feature2', 'Correlation']
    
    # Filter the pairs to show only those with strong correlations
    strong_correlations = correlation_pairs_df[
        (correlation_pairs_df['Correlation'].abs() > threshold) & 
        (correlation_pairs_df['Feature1'] != correlation_pairs_df['Feature2'])
    ]
    
    # Remove duplicate pairs (e.g., (A, B) and (B, A))
    strong_correlations['Pair'] = strong_correlations.apply(
        lambda row: tuple(sorted([row['Feature1'], row['Feature2']])), axis=1
    )
    strong_correlations = strong_correlations.drop_duplicates(subset=['Pair']).drop(columns=['Pair'])
    
    return correlation_matrix, strong_correlations

# Create a function to display the correlation matrix
def show_matrix(matrix, target):
    """
    Function to display the correlation matrix.

    Parameters:
    matrix (pd.DataFrame): DataFrame containing feature correlations.
    target (string): Target feature.

    Returns:
    None
    """
    # Import the matplotlib.pyplot and seaborn libraries
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Visualize the correlation matrix for the dataset minus the target feature and 'win_percentage' using a heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title(f'{target} Correlations')
    plt.xticks(rotation=45, ha='right')
    plt.show()

# Create a function to drop redundant features from the dataset to mitigate high multicollinearity within the model
def drop_redundant_feats(df, df1):
    """
    A function to drop features from a dataset that will cause high multicollinearity in the Lasso regression model.

    Parameters:
    df (pd.DataFrame): DataFrame containing features with high correlations.
    df1 (pd.DataFrame): DataFrame containing features from dataset.

    Returns:
    df_reduced_feats (pd.DataFrame): DataFrame with data from selected features.
    """
    # Create a list of redundant features to drop
    redundant_feats = df['Feature1'].unique()

    # Create a copy of df1 to avoid modifying the original DataFrame
    df_reduced_feats = df1.copy()
    
    # Drop the redundant features from the copied DataFrame
    df_reduced_feats.drop(columns=redundant_feats, inplace=True, errors='ignore')
    
    return df_reduced_feats

# Create a function to determine variance inflation factors
def gen_vif(X):
    """
    Function to determine variance inflation factors.

    Parameters:
    X (pd.DataFrame): DataFrame containing the feature set.

    Returns:
    vif_data (pd.DataFrame): DataFrame containing features and their variance inflation factor scores.
    """
    # Import the needed libraries
    import pandas as pd
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Create a copy of X to preserve the original data
    X_copy = X.copy()
    X_copy['intercept'] = 1
    vif_data = pd.DataFrame()
    vif_data['feature'] = X_copy.columns
    vif_data['vif'] = [variance_inflation_factor(X_copy.values, i) for i in range(X_copy.shape[1])]

    return vif_data

# Create a function to fit and train a lasso model, returning a dictionary of various metrics and an updated feature DataFrame
def gen_lasso_model(feats, y, alpha_grid=None, test_size=0.2, random_state=42):
    """
    Train a Lasso Regression model after finding the best alpha. Returns a dictionary containing the model, performance 
    metrics, non-zero coefficients, feature names, y measurables, and an updated DataFrame with feature data. 
    
    Parameters:
    feats (pd.DataFrame): DataFrame containing feature variables.
    y (pd.Series): Target variable.
    alpha_grid (list or None): List of alpha values to search. If None, default grid is used.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    results (Dictionary): Dictionary containing the model, performance metrics, non-zero coefficients, and feature names.
    updated_df (pd.DataFrame): DataFrame containing updated feature data.
    """
    # Import needed libraries
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.linear_model import Lasso
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler

    # Create an alpha grid
    if alpha_grid is None:
        alpha_grid = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

    y_measurables = y.describe()

    while True:
        # Store the feature names
        feature_names = feats.columns

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feats)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
        
        # Create the grid search object
        lasso_grid = GridSearchCV(Lasso(), {'alpha': alpha_grid}, cv=5, scoring='neg_mean_squared_error')
        
        # Fit the grid to the data
        lasso_grid.fit(X_train, y_train)
        
        # Get the best alpha value
        best_alpha = lasso_grid.best_params_['alpha']
        
        # Retrain the new model using the best alpha score
        lasso = Lasso(alpha=best_alpha)
        lasso.fit(X_train, y_train)

        # Predictions
        y_train_pred = lasso.predict(X_train)
        y_test_pred = lasso.predict(X_test)

        # Performance metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Get coefficients from new model
        coefficients = pd.Series(lasso.coef_, index=feature_names)

        # Calculate and store non-zero coefficients
        non_zero_coefficients = coefficients[coefficients != 0].sort_values(ascending=False)

        # Calculate and store features with coefficients of zero
        zero_coefficients = coefficients[coefficients == 0].index.tolist()

        # Create an updated dataset 
        updated_df = feats.drop(zero_coefficients, axis=1)

        # If there are no zero coefficients, break the loop
        if len(zero_coefficients) == 0:
            break
        
        # Update feats for the next iteration
        feats = updated_df

    # Call the gen_vif function to calculate variance inflation factors
    vif_data = gen_vif(pd.DataFrame(X_scaled, columns=feature_names))

    # Print the best alpha, y measurables, metric scores, non-zero coefficients, and vif data
    print(f'Best Alpha: {best_alpha}')
    print(f'Y Measurables: \n{y_measurables}')
    print(f'Training RMSE: {train_rmse}, Testing RMSE: {test_rmse}')
    print(f'Training R^2: {train_r2}, Testing R^2: {test_r2}')
    print(f'Non-zero Coefficients: \n{non_zero_coefficients}')
    print(f'Variance Inflation Factors: \n{vif_data}')

    # Return the results as a dictionary
    results = {
        'model': lasso,
        'X': X_scaled,
        'y': y,
        'non_zero_coefficients': non_zero_coefficients,
        'feature_names': feature_names
    }

    return results, updated_df

# Create a function to generate a horizontal bar chart displaying the regression coefficients to the target feature
def gen_coeff_barh(coefficients, target):
    """
    Function to generate a horizontal bar chart displaying feature regression coefficients to the target feature.

    Parameters:
    coefficients (pd.Series): Series of features and regression coefficient values.
    target (string): Target feature.

    Returns:
    None
    """
    # Import the needed libraries
    import matplotlib.pyplot as plt

    # Set the height proportionally with the length of coefficients
    height = len(coefficients)
    
    # Visualize coefficients with a horizontal bar chart
    plt.figure(figsize=(12, height))
    coefficients.sort_values().plot(kind='barh')
    plt.title(f'Coefficients for {target}')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.show()

    return

# Create a function to generate bootstrapped regression coefficients and calculate the 95% confidence intervals along with evaluation metrics
def gen_bootstrap_coefficients(model, X, y, feature_names, n_bootstraps=1000):
    """
    Generate bootstrapped regression coefficients and calculate 95% confidence intervals for a Lasso regression model. Returns and prints a DataFrame 
    with the sorted coefficient summary. Also prints average target value and evaluation metrics for the bootstrapped model. 
    
    Parameters:
    model (Lasso): A trained Lasso regression model.
    X (DataFrame): Features data.
    y_train (Series): Target variable.
    feature_names (list): List of feature names.
    n_bootstraps (int): Number of bootstrap samples.
    
    Returns:
    sf_df (DataFrame): DataFrame containing the sorted results.
    """
    # Import the needed libraries
    import pandas as pd
    import numpy as np
    from sklearn.utils import resample
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Lists to store bootstrap coefficients, y means, RMSE, and R-squared scores
    bootstrap_coefficients_list = []
    train_rmse_scores = []
    test_rmse_scores = []
    train_r2_scores = []
    test_r2_scores = []

    # Bootstrap loop
    for _ in range(n_bootstraps):
        X_resampled, y_resampled = resample(X,y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

        model.fit(X_train, y_train)
        
        # Collect each series of coefficients
        bootstrap_coefficients_list.append(pd.Series(model.coef_, index=feature_names))

        # Predict and calculate RMSE and R^2 scores for the training set
        y_train_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse_scores.append(train_rmse)
        train_r2_scores.append(train_r2)

        # Predict and calculate RMSE and R^2 scores for the testing set
        y_test_pred = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse_scores.append(test_rmse)
        test_r2_scores.append(test_r2)
        
    # Concatenate all series stored in the list into a dataframe
    bootstrap_coefficients = pd.concat(bootstrap_coefficients_list, axis=1).transpose()
    
    # Calculate the mean and standard deviation of the bootstrapped coefficients
    coefficient_mean = bootstrap_coefficients.mean()
    coefficient_std = bootstrap_coefficients.std()
    
    # Calculate the 95% confidence intervals for each coefficient
    ci_lower = bootstrap_coefficients.quantile(0.025)
    ci_upper = bootstrap_coefficients.quantile(0.975)
    
    # Create a DataFrame to summarize the results
    coefficient_summary = pd.DataFrame({
        'Mean': coefficient_mean,
        'StdDev': coefficient_std,
        'CI Lower': ci_lower,
        'CI Upper': ci_upper
    })
    
    # Select significant features based on the absolute value of the coefficient mean and distribution
    selected_features = coefficient_summary.index.tolist()

    # Create a DataFrame from the bootstrap results and sort it by the mean coefficient
    sf_df = coefficient_summary.loc[selected_features].sort_values(by='Mean', ascending=False)

    # Calculate and store the average of the target variable
    avg_y = np.mean(y)

    # Calculate and store the mean RMSE and R^2 scores for both training and testing sets
    avg_train_rmse = np.mean(train_rmse_scores)
    avg_test_rmse = np.mean(test_rmse_scores)
    avg_train_r2 = np.mean(train_r2_scores)
    avg_test_r2 = np.mean(test_r2_scores)

    # Print the evaluation metrics and the sorted coefficients
    print(f'Average Y: {avg_y}')
    print(f'Average Training RMSE: {avg_train_rmse}, Average Testing RMSE: {avg_test_rmse}')
    print(f'Average Training R^2: {avg_train_r2}, Average Testing R^2: {avg_test_r2}')
    print(f'Sorted Coefficients: \n{sf_df}')
    
    return sf_df

# Create a function to generate an error bar chart containing regression coefficients with 95% confidence intervals to the target feature
def gen_ebar(bootstrap_results):
    """
    Function to generate and display an error bar chart containing feature regression coefficients with 95% confidence intervals to the target feature.

    Parameters:
    bootstrap_results (pd.DataFrame): DataFrame containing the mean, standard deviation, and lower/upper confidence intervals of the regression 
    coefficients to the target feature.

    Returns:
    None
    """
    # Import the needed libraries
    import matplotlib.pyplot as plt

    # Set the width proportionally with the length of coefficients
    width = len(bootstrap_results)
    
    # Plot the results
    plt.figure(figsize=(width, 6))
    plt.errorbar(
        bootstrap_results.index,
        bootstrap_results['Mean'],
        yerr=[
            bootstrap_results['Mean'] - bootstrap_results['CI Lower'],
            bootstrap_results['CI Upper'] - bootstrap_results['Mean']
        ],
        fmt='o',
        capsize=5
    )
    plt.axhline(0, color='grey', linestyle='--')
    plt.title('Regression Coefficients with 95% Confidence Intervals')
    plt.xlabel('Feature')
    plt.ylabel('Coefficient')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to fit labels
    plt.show()

    return

# Create a function to calculate the effects of the selected features on the target
def effect_on_target(bootstrap_results, df, is_percentage=True):
    """
    Calculate the effect on the target for each feature after bootstrapping occurs and print the results.

    Parameters:
    target (string): Name of target variable.
    bootstrap_results (pd.DataFrame): DataFrame containing feature statistics with columns ['Mean', 'StdDev', 'CI Lower', 'CI Upper'].
    df (pd.DataFrame): Original DataFrame containing features.

    Returns:
    None
    """
    # Import the needed libraries
    import numpy as np
    
    for index, row in bootstrap_results.iterrows():
        ci_lower = row['CI Lower']
        ci_upper = row['CI Upper']

        # Exclude features whose confidence intervals include 0
        if ci_lower <= 0 <= ci_upper:
            continue

        
        feature_name = index
        regression_coefficient = row['Mean']

        # Calculate the increment as 2% of the mean value of the feature
        feature_mean = np.mean(df[feature_name])
        increment = feature_mean * 0.02

        # Get the standard deviation of the original (non-scaled) feature
        feature_std = np.std(df[feature_name])

        # Calculate the scaled feature increment
        delta_x_scaled = increment / feature_std

        # Calculate the effect on the target variable
        delta_y = regression_coefficient * delta_x_scaled

        # Print the effect statement
        if is_percentage:
            print(f'A {increment:.3f} unit increase in {feature_name} results in an approximate {delta_y:.4f} (or {delta_y * 100:.2f} percentage points) change in the target variable.')
        else:
            print(f"A {increment:.3f} unit increase in {feature_name} results in an approximate {delta_y:.3f} change in the target variable.")
