import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb
from bayes_opt import BayesianOptimization
import warnings
import shap
warnings.filterwarnings('ignore')

#################################################################
#################################################################
file_path = './filter_data3.csv'
data = pd.read_csv(file_path, index_col='index')

X = data.iloc[:, 1:]
y = data['ln_abs_c_interface_mean']
cols = X.columns
#################################################################
#################################################################
model_filename = './1_Bmodel.pkl'
model = joblib.load(model_filename)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1200)
shap.summary_plot(
    shap_values, 
    X,
    feature_names=X.columns,
    plot_type="dot",
    show=False,
    color_bar=True,
    cmap='viridis',
)
plt.gca().set_position([0.2, 0.2, 0.65, 0.65])
ax1 = plt.gca()
ax2 = ax1.twiny()
shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    show=False
)
plt.gca().set_position([0.2, 0.2, 0.65, 0.65])
ax2.axhline(
    y=13, 
    color='black',
    linestyle='-',
    linewidth=4
)
bars = ax2.patches
for bar in bars:    
    bar.set_alpha(0.2)
ax1.set_xlabel('SHAP Value Contribution', fontsize=20, fontweight='bold')
ax2.set_xlabel('Mean SHAP Value', fontsize=20, fontweight='bold')
ax2.xaxis.set_label_position('top')
ax2.xaxis.tick_top()
ax1.set_ylabel('Features', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.gca().spines['bottom'].set_linewidth(2.5)
ax1.tick_params(axis='both', which='major', labelsize=20, width=2.5)
ax2.tick_params(axis='x', which='major', labelsize=20, width=2.5)

plt.savefig(
    "SHAP.jpg",
    format='jpg',
    bbox_inches='tight'
)
