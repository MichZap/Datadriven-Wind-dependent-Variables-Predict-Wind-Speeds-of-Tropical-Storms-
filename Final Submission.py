#input for model 1 and model 2
Model1_submission=r'G:\HU\Data\submission-r3d-past-3avg-02-01 22-21-42.csv'
Model2_submission=r'G:\HU\Data\submission-effnetb0-3ch-past-3rot-9h-02-01 22-21-42.csv'

#code
import pandas as pd
from datetime import datetime

df1=pd.read_csv(Model1_submission)
df2=pd.read_csv(Model2_submission)

df_comb=df1.merge(df2,how="left",on="image_id")
df_comb["Prediction"]=df_comb.drop(["image_id"],axis=1).mean(axis=1)

DATA_PATH =r'G:\HU\Data'

#submission format
submission_format = pd.read_csv(
    DATA_PATH + "/submission_format.csv", index_col="image_id"
)

df_comb["wind_speed"]=round(df_comb["Prediction"]).astype(int)
submission_frame=df_comb[["image_id","wind_speed"]].set_index("image_id")



today = datetime.now()
submission_frame.to_csv((DATA_PATH + "/submission_best2_mix_3rot"+today.strftime("%Y-%m-%d %H-%M-%S")+".csv"), index=True)
