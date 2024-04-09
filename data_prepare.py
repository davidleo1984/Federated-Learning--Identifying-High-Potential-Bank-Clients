import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


# 模拟银行客户的数据 
# a function to simulate data of customers in a branch
def simulate_data(n=100, seed=42):
    np.random.seed(seed)

    # 人口统计信息 
    # demographic information
    ages = np.random.randint(18, 70, size=n)
    genders = np.random.choice(['Male', 'Female'], size=n)
    occupations = np.random.choice(['Salaried', 'Self-Employed', 'Retired', 'Student'], size=n)
    
    account_balances = np.zeros(n)
    transaction_histories = np.zeros(n)
    loan_histories = np.zeros(n)

    # 根据不同的人口统计特征调整账户信息的分布
    # adjust account information based on demographic features
    for i in range(n):
        if occupations[i] == 'Salaried':
            account_balances[i] = np.random.uniform(5000, 75000)
            transaction_histories[i] = np.random.uniform(500, 7500)
            loan_histories[i] = np.random.uniform(0, 20000)
        elif occupations[i] == 'Self-Employed':
            account_balances[i] = np.random.uniform(10000, 100000)
            transaction_histories[i] = np.random.uniform(1000, 10000)
            loan_histories[i] = np.random.uniform(0, 25000)
        elif occupations[i] == 'Retired':
            account_balances[i] = np.random.uniform(20000, 80000)
            transaction_histories[i] = np.random.uniform(100, 3000)
            loan_histories[i] = np.random.uniform(0, 15000)
        else:  # Students
            account_balances[i] = np.random.uniform(1000, 20000)
            transaction_histories[i] = np.random.uniform(100, 5000)
            loan_histories[i] = np.random.uniform(0, 10000)
    
    data = pd.DataFrame({
        'Age': ages,
        'Gender': genders,
        'Occupation': occupations,
        'Account Balance': account_balances,
        'Monthly Transactions': transaction_histories,
        'Monthly Loan Repayment': loan_histories
    })
    
    return data

# 为数据集添加“高潜力”标签
# a function to add a "High Potential" label to each customer in the dataset
def add_label(df):
    
    # 定义高潜力客户筛选规则
    # criteria for high potential customers
    criteria_1 = df['Account Balance'] > df['Account Balance'].mean()
    criteria_2 = df['Monthly Transactions'] > df['Monthly Transactions'].mean()
    criteria_3 = df['Monthly Loan Repayment'] < df['Monthly Loan Repayment'].mean()
    
    df['High Potential'] = ((criteria_1 & criteria_2) | (criteria_1 & criteria_3) | (criteria_2 & criteria_3)).astype(int)
    return df

# 自定义数据集类
# A custom dataset class
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.values
        self.y = y.values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_data(n, simulate_seed=42, split_seed=42, batch_size=16):
    df = simulate_data(n=n, seed=simulate_seed)
    df_labeled = add_label(df)

    # 对分类变量进行One-Hot编码
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df_labeled, columns=['Gender', 'Occupation'], dtype='float32')

    # 将特征和目标变量分开
    # Separate features and target variable
    features = df_encoded.drop('High Potential', axis=1)
    target = df_encoded['High Potential']

    # 划分数据集
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=split_seed)
    
    # 创建训练和测试数据集
    # Create training and testing datasets
    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)

    # 创建数据加载器
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_examples = {"trainset" : len(train_dataset), "testset" : len(test_dataset)}

    return train_loader, test_loader, num_examples