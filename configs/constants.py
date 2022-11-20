import os


DATA_ROOT = os.path.abspath(os.path.join(__file__, "../..", "data"))

# MODEL
SUPPORT_MODEL = (
    "LogisticRegression",
    "DecisionTreeClassifier",
    "MLPClassifier",
    "RandomForestClassifier",
    "LGBMClassifier",
    "XGBClassifier",
    "GradientBoostingClassifier"
)
DEFAULT_MODEL = "GradientBoostingClassifier"
