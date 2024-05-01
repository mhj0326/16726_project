import os
import pandas as pd

#WORK_DIR = "/path/to/load/vindr-cxr"
WORK_DIR = "/home/rwiddhi/rwiddhi/cxr-clip/"
df = pd.read_csv(os.path.join(WORK_DIR, "image_labels_test.csv"))

df = df.groupby("image_id").agg(sum)
df.loc[:, "Aortic enlargement":"No finding"] = (df.loc[:, "Aortic enlargement":"No finding"] > 0).astype(int)
df.reset_index(drop=False, inplace=True)

#df_train, df_valid = train_test_split(df, test_size=0.2, random_state=0)

#df_train.to_csv(os.path.join(WORK_DIR, "vindr_train.csv"))
#df_valid.to_csv(os.path.join(WORK_DIR, "vindr_valid.csv"))
df.to_csv(os.path.join(WORK_DIR, "vindr_test.csv"))