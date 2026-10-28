from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'raw'
CLEAN = ROOT / 'data' / 'clean'
DB = ROOT / 'data' / 'instamart.db'
CLEAN.mkdir(parents=True, exist_ok=True)

# Read raw exports without modifying the source files.
customers = pd.read_csv(RAW / 'customers.csv', keep_default_na=False)
products = pd.read_csv(RAW / 'products.csv')
orders = pd.read_csv(RAW / 'orders.csv', keep_default_na=False)

# Customers: standardize city; normalize gender; impute invalid/missing ages by city median.
customers['city'] = customers['city'].astype(str).str.strip().str.title()
gender = customers['gender'].astype(str).str.strip().str.lower()
customers['gender'] = gender.map({
    'male': 'male', 'm': 'male', 'female': 'female', 'f': 'female'
}).fillna('unknown')
customers['age'] = pd.to_numeric(customers['age'].replace('', pd.NA), errors='coerce')
customers.loc[~customers['age'].between(10, 100), 'age'] = pd.NA
customers['age'] = customers['age'].fillna(customers.groupby('city')['age'].transform('median'))
customers['age'] = customers['age'].round().astype('int64')
customers['signup_date'] = pd.to_datetime(customers['signup_date'], errors='coerce')

# Products: retain supplied catalogue and enforce numeric price.
products['price_inr'] = pd.to_numeric(products['price_inr'], errors='raise')

# Orders: parse timestamp, remove exact duplicates, then keep the latest record per order.
orders['order_ts'] = pd.to_datetime(orders['order_ts'], errors='coerce')
orders = orders.drop_duplicates()
orders = orders.sort_values(['order_id', 'order_ts'], kind='mergesort')
orders = orders.drop_duplicates(subset='order_id', keep='last').copy()
orders['quantity'] = pd.to_numeric(orders['quantity'], errors='raise').astype(int)
orders['discount_pct'] = pd.to_numeric(orders['discount_pct'], errors='raise')
orders['order_amount'] = pd.to_numeric(orders['order_amount'], errors='coerce')
orders['delivery_minutes'] = pd.to_numeric(orders['delivery_minutes'].replace('', pd.NA), errors='coerce')
orders.loc[orders['delivery_minutes'] < 0, 'delivery_minutes'] = pd.NA

# Recover unparseable amounts from catalogue price x quantity x discount.
orders = orders.merge(products[['product_id', 'price_inr', 'category']], on='product_id', how='left', validate='many_to_one')
missing_amount = orders['order_amount'].isna()
orders.loc[missing_amount, 'order_amount'] = (
    orders.loc[missing_amount, 'price_inr'] * orders.loc[missing_amount, 'quantity'] *
    (1 - orders.loc[missing_amount, 'discount_pct'] / 100)
).round(2)

# Impute only completed orders, using category medians from completed orders.
completed_mask = orders['status'].eq('completed')
category_medians = orders.loc[completed_mask].groupby('category')['delivery_minutes'].transform('median')
orders.loc[completed_mask, 'delivery_minutes'] = orders.loc[completed_mask, 'delivery_minutes'].fillna(category_medians)
# Keep cancelled delivery times missing: they were not delivered.
orders.loc[orders['status'].eq('cancelled'), 'delivery_minutes'] = pd.NA
orders['order_amount'] = orders['order_amount'].round(2)
orders['delivery_minutes'] = orders['delivery_minutes'].round().astype('Int64')
orders['order_ts'] = orders['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
customers['signup_date'] = customers['signup_date'].dt.strftime('%Y-%m-%d')

# Remove helper columns before saving the orders table.
orders = orders.drop(columns=['price_inr', 'category'])

for name, df in [('customers', customers), ('products', products), ('orders', orders)]:
    df.to_csv(CLEAN / f'{name}.csv', index=False)

DB.parent.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(DB) as con:
    for name, df in [('customers', customers), ('products', products), ('orders', orders)]:
        df.to_sql(name, con, if_exists='replace', index=False)

print(f'Cleaned customers={len(customers)}, products={len(products)}, orders={len(orders)}')
print('Completed order amount total INR {:.2f}'.format(orders.loc[orders.status.eq('completed'), 'order_amount'].sum()))
