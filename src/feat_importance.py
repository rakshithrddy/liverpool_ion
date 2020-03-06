import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import joblib
warnings.simplefilter(action='ignore', category=FutureWarning)

class FeatImportance:
    def __init__(self):




def plotImplgb(model, X , num = 90):
    feature_imp = pd.DataFrame({'Value':model.feature_importance(),'Feature':X.columns})
    plt.figure(figsize=(40, 90))
    sns.set(font_scale = 5)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                        ascending=False)[0:num])
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()
    return feature_imp





# importance = []
# non_importance = []
# columns = train_df.columns
# for cols in columns:
#   if cols not in importance_fet:
#     importance.append(cols)
#   else:
#     non_importance.append(cols)
#
# print(len(importance))
# print(non_importance)