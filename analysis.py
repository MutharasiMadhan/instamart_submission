from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
with sqlite3.connect(ROOT / 'data' / 'instamart.db') as con:
    orders = pd.read_sql_query('SELECT * FROM orders', con)
    customers = pd.read_sql_query('SELECT * FROM customers', con)
    products = pd.read_sql_query('SELECT * FROM products', con)

orders['order_amount'] = pd.to_numeric(orders['order_amount'])
orders['discount_pct'] = pd.to_numeric(orders['discount_pct'])
orders['quantity'] = pd.to_numeric(orders['quantity'])
orders['delivery_minutes'] = pd.to_numeric(orders['delivery_minutes'], errors='coerce')
completed = orders.loc[orders.status.eq('completed')].copy()

# Q2 recreated as a pivot table with margins.
joined = completed.merge(customers[['customer_id', 'city']], on='customer_id').merge(
    products[['product_id', 'category']], on='product_id')
pivot = pd.pivot_table(joined, index='city', columns='category', values='order_amount',
                       aggfunc='sum', fill_value=0, margins=True, margins_name='All')
print('\n=== Q2 PIVOT TABLE (INR) ===')
print(pivot.round(2).to_string())
print('Pivot grand total INR {:.2f}'.format(pivot.loc['All', 'All']))

# Named aggregation per city.
city_stats = completed.merge(customers[['customer_id', 'city']], on='customer_id').groupby('city').agg(
    total_order_amount=('order_amount', 'sum'),
    mean_order_amount=('order_amount', 'mean'),
    completed_order_count=('order_amount', 'count'))
print('\n=== COMPLETED SPEND BY CITY ===')
print(city_stats.round(2).to_string())

# Distribution and Tukey outliers.
s = completed['order_amount'].dropna()
q1, median, q3 = s.quantile([.25, .50, .75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print('\n=== COMPLETED ORDER AMOUNT DISTRIBUTION (INR) ===')
print(f'count={s.count()} mean={s.mean():.2f} median={median:.2f} skewness={s.skew():.2f}')
print(f'min={s.min():.2f} Q1={q1:.2f} Q2={median:.2f} Q3={q3:.2f} max={s.max():.2f}')
print(f'IQR={iqr:.2f} lower_fence={lower:.2f} upper_fence={upper:.2f}')
print(f'outliers below lower fence={(s < lower).sum()}')
print(f'outliers above upper fence={(s > upper).sum()}')

cols = ['order_amount', 'discount_pct', 'quantity', 'delivery_minutes']
corr = completed[cols].corr(numeric_only=True)
print('\n=== CORRELATION MATRIX ===')
print(corr.round(3).to_string())
abs_corr = corr.abs().where(~np.eye(len(corr), dtype=bool))
a, b = abs_corr.stack().idxmax()
print(f'Largest absolute correlation: {a} vs {b}: {corr.loc[a, b]:.3f}')

# Compact narrative <=200 words, written from the measured results.
print('\n=== EXPLORATORY SUMMARY ===')
print(
    f'This dataset describes Instamart customers, products, and orders across six Indian cities. '
    f'There are {len(completed):,} completed orders. Order amounts are right-skewed: the mean is INR {s.mean():.2f}, '
    f'while the median is INR {median:.2f}; skewness is {s.skew():.2f}. The middle half runs from '
    f'INR {q1:.2f} to INR {q3:.2f}. Using the 1.5×IQR rule, {int((s > upper).sum())} orders exceed '
    f'the upper fence of INR {upper:.2f}, while {int((s < lower).sum())} fall below the lower fence. '
    f'These may reflect larger baskets or higher-priced products and warrant review rather than automatic deletion. '
    f'The strongest absolute correlation among the requested variables is {a} with {b} '
    f'(r={corr.loc[a, b]:.3f}). This is a weak association; correlation of this size does not establish causation.'
)
