import pandas as pd

# 提取银行数据的PSI tokens
df_bank = pd.read_csv('partyA_bank.csv')
bank_tokens = df_bank['psi_token'].tolist()

with open('bank_tokens.txt', 'w') as f:
    f.write('\n'.join(bank_tokens))

print(f'银行数据PSI tokens已保存到bank_tokens.txt，共{len(bank_tokens)}个')

# 提取电商数据的PSI tokens
df_ecom = pd.read_csv('partyB_ecom.csv')
ecom_tokens = df_ecom['psi_token'].tolist()

with open('ecom_tokens.txt', 'w') as f:
    f.write('\n'.join(ecom_tokens))

print(f'电商数据PSI tokens已保存到ecom_tokens.txt，共{len(ecom_tokens)}个')