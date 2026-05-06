import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/raw/race.csv')
print(df.shape)
print(df.columns.tolist())
print(df.head(2))
train_val_df, test_df = train_test_split(
    df, 
    test_size=0.10,     
    random_state=42,      
    shuffle=True
)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.111,     
    random_state=42,
    shuffle=True
)

print(f"Train size:      {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size:       {len(test_df)}")
train_df.to_csv('data/raw/train.csv', index=False)
val_df.to_csv('data/raw/val.csv',   index=False)
test_df.to_csv('data/raw/test.csv', index=False)

print("Splits saved successfully!")