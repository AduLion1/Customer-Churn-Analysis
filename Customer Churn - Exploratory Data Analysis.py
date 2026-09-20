# Customer Churn - Exploratory Data Analysis (EDA)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 0. LOAD DATA
df = pd.read_csv(r"C:\Users\adubi\OneDrive\Desktop\Data Analytic Project\customer_churn_dataset-testing-master.csv")

# 1. STRUCTURAL CHECK (confirms cleaning carried over)
print("="*60)
print("STRUCTURAL CHECK")
print("="*60)
df.info()
print("\nDescribe:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

# 2. OVERALL CHURN RATE
print("\n" + "="*60)
print("OVERALL CHURN RATE")
print("="*60)
churn_rate = df['Churn'].mean()
print(f"Overall churn rate: {churn_rate:.1%}")

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Churn')
plt.title('Churn Distribution')
plt.tight_layout()
plt.show()

# 3. CHURN RATE BY CATEGORICAL VARIABLES
print("\n" + "="*60)
print("CHURN RATE BY CATEGORICAL VARIABLES")
print("="*60)
for col in ['Subscription Type', 'Contract Length', 'Gender']:
    rate = df.groupby(col)['Churn'].mean().sort_values(ascending=False)
    print(f"\nChurn rate by {col}:\n{rate}")

    plt.figure(figsize=(6,4))
    sns.barplot(data=df, x=col, y='Churn')
    plt.title(f'Churn Rate by {col}')
    plt.ylabel('Churn Rate')
    plt.tight_layout()
    plt.show()

# 4. CHURN VS NUMERIC VARIABLES (distribution plots)
print("\n" + "="*60)
print("CHURN VS NUMERIC VARIABLES")
print("="*60)
numeric_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls',
                 'Payment Delay', 'Total Spend', 'Last Interaction']

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(data=df, x=col, hue='Churn', kde=True, multiple='stack')
    plt.title(f'{col} Distribution: Churned vs Retained')
    plt.tight_layout()
    plt.show()

# 5. CORRELATION MATRIX
print("\n" + "="*60)
print("CORRELATION MATRIX")
print("="*60)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# 6. STATISTICAL TESTS
print("\n" + "="*60)
print("STATISTICAL TESTS")
print("="*60)

# Chi-square: Contract Length vs Churn
contingency = pd.crosstab(df['Contract Length'], df['Churn'])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nContract Length vs Churn — Chi2={chi2:.2f}, p={p:.4f}")

# Chi-square: Subscription Type vs Churn
contingency2 = pd.crosstab(df['Subscription Type'], df['Churn'])
chi2b, pb, dofb, expectedb = stats.chi2_contingency(contingency2)
print(f"Subscription Type vs Churn — Chi2={chi2b:.2f}, p={pb:.4f}")

# t-tests: churned vs retained, for each key numeric variable
print("\nT-tests (churned vs retained):")
for col in ['Support Calls', 'Payment Delay', 'Tenure', 'Total Spend', 'Usage Frequency']:
    churned = df[df['Churn'] == 1][col]
    retained = df[df['Churn'] == 0][col]
    t_stat, p_val = stats.ttest_ind(churned, retained)
    print(f"\n{col}:")
    print(f"   t={t_stat:.2f}, p={p_val:.4f}")
    print(f"   Mean churned: {churned.mean():.2f} | Mean retained: {retained.mean():.2f}")

print("\n" + "="*60)
print("EDA COMPLETE — review charts and printed stats above")
print("="*60)