import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

TEST_SIZE = .10  # defining portion of data to be part of testing
VAL_SIZE = 0.10  # defining portion of training data to be part of validation

# train_data_m contain all the images name and target_labels_ids
raw_data = pd.read_csv('data/external/train_data_m.csv')
image_ids = raw_data['path'].to_numpy()
labels = raw_data['labels'].to_numpy()

# splitting data in train validation and testing
X_train_val, X_test, y_train_val, y_test = train_test_split(
    image_ids, labels, test_size=TEST_SIZE, random_state=42)

# creating dataset for train_validation
train_val_df = pd.concat(
    [pd.Series(X_train_val), pd.Series(y_train_val)], axis=1)
train_val_df.to_csv('data/interim/train_val_data.csv', index=False)

# creating dataset for testing
test_df = pd.concat([pd.Series(X_test), pd.Series(y_test)], axis=1)
test_df.to_csv('data/processed/test_data.csv', index=False)

# splitting data from training and validation
files_names = train_val_df[0]
train_val_labels = train_val_df[1]

# splitting data in train validation and testing
X_train, X_val, y_train, y_val = train_test_split(files_names,
                                                  train_val_labels,
                                                  test_size=VAL_SIZE,
                                                  random_state=66)

# training data
do_over_sampling = False

if do_over_sampling:
    over_sampler = RandomOverSampler(sampling_strategy='minority')
    X_train = X_train.to_numpy().reshape(-1,1)
    y_train = y_train.to_numpy()
    X_resampled, y_resampled = over_sampler.fit_resample(X_train, y_train)
    train_df = pd.concat([pd.Series(X_resampled.squeeze()), pd.Series(y_resampled)], axis=1)
else:
    train_df = pd.concat([pd.Series(X_train), pd.Series(y_train)], axis=1)

train_df.to_csv('data/processed/train_data.csv', index=False)

# validation data
val_df = pd.concat([pd.Series(X_val), pd.Series(y_val)], axis=1)
val_df.to_csv('data/processed/val_data.csv', index=False)
