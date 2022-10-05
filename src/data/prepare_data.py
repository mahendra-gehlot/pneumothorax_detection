import pandas as pd
from sklearn.model_selection import train_test_split


TEST_SIZE = .20  # defining portion of data to be part of testing

# train_data_m contain all the images name and target_labels_ids
raw_data = pd.read_csv('data/external/train_data_m.csv')
image_ids = raw_data['file_name'].to_numpy()
labels = raw_data['target'].to_numpy()

# splitting data in train validation and
X_train_val, X_test, y_train_val, y_test = train_test_split(image_ids,
                                                            labels,
                                                            test_size=0.20,
                                                            random_state=42)

# creating dataset for train_validation
train_val_df = pd.concat([pd.Series(X_train_val), pd.Series(y_train_val)], axis=1)
train_val_df.to_csv('data/interim/train_val_data.csv', index=False)

# creating dataset for testing
test_df = pd.concat([pd.Series(X_test), pd.Series(y_test)], axis=1)
test_df.to_csv('data/interim/test_data.csv', index=False)
