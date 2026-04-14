import pandas as pd
import random
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# 设置随机种子
random.seed(42)
np.random.seed(42)

file_path = r"D:your document.xlsx"
df = pd.read_excel(file_path)

# using stratified group cross-validation
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
split = next(sgkf.split(X=df, y=df['label'], groups=df['user name']))

train_idx, test_idx = split
test_df = df.iloc[test_idx]
train_df = df.iloc[train_idx]

# Automatically verify no user overlap
def validate_integrity(test, train):
    common = set(test['user name']).intersection(set(train['user name']))
    assert len(common) == 0, f"Users affected by the data breach: {common}"
    print("✅ User integrity validation passed")

validate_integrity(test_df, train_df)

# Verify the distribution of labels
orig_ratio = df['label'].mean()
test_ratio = test_df['label'].mean()
train_ratio = train_df['label'].mean()

print(f"\n原始标签比例: {orig_ratio:.4f}")
print(f"训练集标签比例: {train_ratio:.4f}")
print(f"测试集标签比例: {test_ratio:.4f}")

if abs(test_ratio - orig_ratio) >= 0.05:
    print("⚠️ 警告：测试集标签分布与原始分布偏差超过5%（可能由分组限制导致）")
else:
    print("✅ 标签分布验证通过")

# 保存结果
test_df.to_excel(r"D:\your document", index=False)
train_df.to_excel(r"D:\your document.xlsx", index=False)