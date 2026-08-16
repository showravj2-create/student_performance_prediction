Student Performance Modeling: Predicting Final Mathematics Grades
Overview
This project investigates whether demographic, family, school, behavioral, and study-related characteristics can explain variation in students' final mathematics grades.
The analysis is designed as a research-oriented machine learning experiment, emphasizing experimental design, reproducibility, model comparison, and critical interpretation rather than simply maximizing a prediction score.
Research Question
How well can final mathematics performance be predicted from student characteristics when intermediate assessment grades are excluded from the feature set?
The target is G3, the final mathematics grade.
G1 and G2 are intentionally excluded. These represent earlier-period grades and are strongly associated with G3. Excluding them makes the prediction task more difficult but better aligned with the question of whether student/background characteristics alone provide predictive information.
Dataset
The project uses the Student Performance dataset from the UCI Machine Learning Repository.
The mathematics dataset contains 395 students and 33 variables, including:
demographic characteristics
family background
school information
study habits
social activities
alcohol consumption indicators
absences
final mathematics grade
Source:
https://archive.ics.uci.edu/dataset/320/student%2Bperformance
Citation:
Cortez, P. (2008). Student Performance. UCI Machine Learning Repository. DOI: 10.24432/C5TG7T.
Methodology
1. Exploratory analysis
The notebook examines:
final-grade distribution
study time and final grades
Spearman correlations
relationships among numerical and ordinal variables
2. Feature design
The target is:
G3
The following are deliberately excluded:
G1
G2
The remaining predictors represent student characteristics and contextual variables.
Categorical variables are one-hot encoded, while numerical variables are median-imputed and standardized where appropriate.
3. Models
Three regression approaches are compared:
Linear Regression — interpretable baseline
Random Forest Regressor — nonlinear ensemble model
Support Vector Regression (SVR) — nonlinear kernel-based model
4. Evaluation
Model selection is based on 5-fold cross-validation performed only on the training set.
Reported metrics:
MAE
RMSE
R²
A final evaluation is then performed on an untouched 20% test set.
The notebook also includes:
predicted-vs-actual diagnostics
residual analysis
permutation importance
Why exclude G1 and G2?
This is a deliberate methodological decision.
G1 and G2 are earlier-period grades for the same subject. They are highly informative about G3, but including them changes the problem from predicting final performance from student/contextual characteristics to predicting a final grade from previous grades.
The UCI documentation explicitly notes the strong relationship between G1, G2, and G3.
Therefore, this project focuses on the more challenging question of whether background and behavioral characteristics contain useful predictive information.
Interpretation
The model outputs should be interpreted as predictive associations, not causal effects.
For example, if study time is associated with higher predicted performance, that does not establish that increasing study time alone would necessarily cause an equivalent increase in final grades.
The dataset is observational and represents students from two Portuguese secondary schools, so external validity is limited.
Limitations
The mathematics dataset contains 395 observations.
The data come from two schools and may not generalize to other educational systems.
Academic performance is influenced by factors that are not captured in the dataset.
A single held-out test set can produce uncertain estimates in a relatively small dataset.
Feature importance does not establish causality.
Excluding G1/G2 makes the task more meaningful for this research question but also substantially harder.
Future Research
Possible extensions include:
repeated/nested cross-validation
uncertainty intervals around model performance
external validation on an independent cohort
SHAP-based model explanations
subgroup error analysis
fairness analysis across demographic groups
classification of students into performance-risk categories
comparison with the Portuguese-language dataset from the same UCI collection
longitudinal modeling if multiple assessment periods are available
Repository Structure
student_performance_prediction/
│
├── data/
│   └── student-mat.csv
│
├── notebooks/
│   └── student_performance_modeling.ipynb
│
├── results/
│   └── figures/
│
├── src/
│   └── train.py
│
├── README.md
└── requirements.txt
Reproducibility
Install dependencies:
pip install -r requirements.txt
Run the notebook from the repository root, or execute:
python src/train.py
Author
Saurabh Das
BSc in Mathematics | Python | Machine Learning | Data Analysis
GitHub: https://github.com/showravj2-create
---

## 🛠 Tech Stack
- **Python 3.10+**
- Libraries:  
  - [NumPy](https://numpy.org/)  
  - [Pandas](https://pandas.pydata.org/)  
  - [Scikit-learn](https://scikit-learn.org/)  
  - [Matplotlib](https://matplotlib.org/)  
  - [Seaborn](https://seaborn.pydata.org/)  

---

## 📂 Repository Structure
