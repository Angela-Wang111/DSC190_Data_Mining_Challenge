"""
data mining challenge for dsc 190
"""

# load packages 
import numpy as np
import pandas as pd 

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import lightgbm as lgbm


# ---------------- aux --------------

def read_train_data() -> pd.DataFrame:
    return pd.read_csv('train.csv')

def read_test_data() -> pd.DataFrame:
    return pd.read_csv('test.csv')

# ---------------- ML ----------------

def date_time_helper(col, year = True):
    """
    Helper function to convert date_time string into year or month column, fillna with most common

    Return cleaned column
    """
    if year:
        transformed = col.fillna('0-0-0').apply(lambda x: int(x.split('-')[0]))
    else:
        transformed = col.fillna('0-0-0').apply(lambda x: int(x.split('-')[1]))
    most_common = transformed.mode()[0]
    fill_complete = transformed.replace(0, most_common)
    return fill_complete


def get_X(train_df: pd.DataFrame, test_df): 
    """ 
    engineer features for train and test df 

    OneHotEncode Features filled with most frequent values
    """

    # Fill NaN for OneHot: training data
    one_hot_columns= ['host_location', 'host_response_time', 'host_neighbourhood', 
    'neighbourhood_cleansed', 'neighbourhood_group_cleansed',  'city', 'state', 'market', 
    'country_code', 'country', 'property_type', 'room_type', 'bed_type', 'cancellation_policy']
    imp_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputed = imp_frequent.fit_transform(train_df[one_hot_columns].to_numpy())
    test_imputed = imp_frequent.fit_transform(test_df[one_hot_columns].to_numpy())
    # one-hot 
    ohe = OneHotEncoder(handle_unknown = 'ignore')
    
    ohe.fit(imputed)
    transformed_train_np = ohe.transform(imputed)
    transformed_test_np = ohe.transform(test_imputed)

    ####### Numerical data below #######
    
    # Transform date_time data
    train_df['host_since_year'] = date_time_helper(train_df['host_since'])
    train_df['host_since_month'] = date_time_helper(train_df['host_since'], year = False)
    
    train_df['first_review_year'] = date_time_helper(train_df['first_review'])
    train_df['first_review_month'] = date_time_helper(train_df['first_review'], year = False)

    train_df['last_review_year'] = date_time_helper(train_df['last_review'])
    train_df['last_review_month'] = date_time_helper(train_df['last_review'], year = False)

    ### test data ###
    
    test_df['host_since_year'] = date_time_helper(test_df['host_since'])
    test_df['host_since_month'] = date_time_helper(test_df['host_since'], year = False)

    test_df['first_review_year'] = date_time_helper(test_df['first_review'])
    test_df['first_review_month'] = date_time_helper(test_df['first_review'], year = False)

    test_df['last_review_year'] = date_time_helper(test_df['last_review'])
    test_df['last_review_month'] = date_time_helper(test_df['last_review'], year = False)

    # Transform host_response_rate column, strip %, and fillna with old mean
    train_df['host_response_rate'] = train_df['host_response_rate'].fillna('-1%').apply(lambda x: int(x.strip('%')))
    host_response_rate = train_df['host_response_rate']
    mean_rate_prev = host_response_rate[host_response_rate.values != -1].mean()
    train_df['host_response_rate'] = train_df['host_response_rate'].replace(-1, mean_rate_prev)

    ### test data ###
    test_df['host_response_rate'] = test_df['host_response_rate'].fillna('-1%').apply(lambda x: int(x.strip('%')))
    host_response_rate = test_df['host_response_rate']
    mean_rate_prev = host_response_rate[host_response_rate.values != -1].mean()
    test_df['host_response_rate'] = test_df['host_response_rate'].replace(-1, mean_rate_prev)


    # Transform extra_people column, strip $
    train_df['extra_people'] = train_df['extra_people'].apply(lambda x: float(x.strip('$')))

    ### test data ###
    test_df['extra_people'] = test_df['extra_people'].apply(lambda x: float(x.strip('$')))

    # Transform binary columns, fillna with majority value
    binary_columns = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
    'instant_bookable', 'require_guest_profile_picture', 'require_guest_phone_verification']
    for col in binary_columns:
        majority = train_df[col].value_counts().idxmax()
        train_df[col] = train_df[col].fillna(majority)
        train_df[col] = train_df[col].apply(lambda x: 1 if (x == 't') else 0)

    ### test data ###
    for col in binary_columns:
        majority = test_df[col].value_counts().idxmax()
        test_df[col] = test_df[col].fillna(majority)
        test_df[col] = test_df[col].apply(lambda x: 1 if (x == 't') else 0)

    # Other numerical columns, fillna or log processed

    numerical_columns = ['host_listings_count', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'guests_included', 
    'minimum_nights', 'maximum_nights', 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 
    'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 
    'review_scores_location', 'review_scores_value', 'calculated_host_listings_count', 'reviews_per_month']

    for col_name in numerical_columns:
        train_df[col_name].fillna(value=train_df[col_name].mean(), inplace=True)
    
    train_df['minimum_nights'] = train_df['minimum_nights'].apply(lambda x: np.log(x + 1))
    train_df['maximum_nights'] = train_df['maximum_nights'].apply(lambda x: np.log(x + 1))


    ### test data ###
    for col_name in numerical_columns:
        test_df[col_name].fillna(value=test_df[col_name].mean(), inplace=True)
    
    test_df['minimum_nights'] = test_df['minimum_nights'].apply(lambda x: np.log(x + 1))
    test_df['maximum_nights'] = test_df['maximum_nights'].apply(lambda x: np.log(x + 1))


    # concat 
    all_num_columns = ['host_since_year', 'host_since_month', 'first_review_year', 
    'first_review_month', 'last_review_year', 'last_review_month'] + ['host_response_rate'] + ['extra_people'] + binary_columns + numerical_columns

    # For training data
    num_train_df = train_df[all_num_columns]
    one_hot_df_train = pd.DataFrame(transformed_train_np.todense(), index = train_df.index)
    final_array_train = one_hot_df_train.merge(num_train_df, left_index = True, right_index = True).to_numpy()

    # For test data
    num_test_df = test_df[all_num_columns]
    one_hot_df_test = pd.DataFrame(transformed_test_np.todense(), index = test_df.index)
    final_array_test = one_hot_df_test.merge(num_test_df, left_index = True, right_index = True).to_numpy()
    
    return (final_array_train, final_array_test)

def get_y(df: pd.DataFrame) -> np.ndarray:
    """ 
    fetch the prediction goal 
    """
    return df['price'].to_numpy()
    
def train(X, y):
    """ train process """
    # list of essential ones: max_depth, n_estimators
    params = {
        'max_depth': 40,
        'n_estimators': 400, 
        'n_jobs': -1
    }

    model = lgbm.LGBMRegressor(**params)
    model.fit(X, y)
    return model
    


# -------------------------------------

def main():
    """
    main running script 
    """
    # load data 
    train_df, test_df = read_train_data(), read_test_data()

    # Process training and testing data
    train_X_processed, test_X_processed = get_X(train_df, test_df)

    # Training
    train_y = get_y(train_df)
    rgb_model = train(train_X_processed, train_y)

    # Prediction
    predicted = rgb_model.predict(test_X_processed)
    print('Prediction Success!')

    # rmse_l = []
    # for _ in range(5):
    #     X_train, X_test, y_train, y_test = train_test_split(train_df.drop(columns = ['price']), 
    #                                                         train_df['price'], test_size=0.25)
    #     train_X_processed, test_X_processed = get_X(X_train, X_test)
    #     rgb_model = train(train_X_processed, y_train)
    #     predicted = rgb_model.predict(test_X_processed)
    #     rmse = mean_squared_error(y_test, predicted, squared=False)
    #     rmse_l.append(rmse)
    # print(rmse_l)
    # print(np.mean(rmse_l))

    # I/O to csv
    result = pd.DataFrame()
    result['Id'] = test_df['id']
    result['Predicted'] = predicted
    result.to_csv('my_test8.csv', index = False)
    
    print('Successfully output csv file!!')


if __name__ == '__main__':
    main()
