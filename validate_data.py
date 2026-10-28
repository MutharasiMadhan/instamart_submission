from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / 'data' / 'clean'
customers = pd.read_csv(CLEAN / 'customers.csv')
orders = pd.read_csv(CLEAN / 'orders.csv')
products = pd.read_csv(CLEAN / 'products.csv')
checks = []
def check(label, condition):
    assert bool(condition), f'FAILED: {label}'
    checks.append(label)

check('six standardized cities', customers.city.nunique() == 6)
check('customer city counts', customers.city.value_counts().to_dict() == {
    'Pune': 227, 'Delhi': 210, 'Bengaluru': 197, 'Chennai': 191, 'Hyderabad': 191, 'Mumbai': 184})
check('normalized gender counts', customers.gender.value_counts().to_dict() == {'female': 610, 'male': 560, 'unknown': 30})
check('no missing customer ages', customers.age.notna().all())
check('customer ages between 10 and 100', customers.age.between(10, 100).all())
check('signup_date stored as ISO date text', customers.signup_date.astype(str).str.match(r'^\d{4}-\d{2}-\d{2}$').all())
check('one order row per order_id', len(orders) == 12309 and orders.order_id.nunique() == 12309)
check('no pending orders', not orders.status.eq('pending').any())
check('expected final statuses', orders.status.value_counts().to_dict() == {'completed': 11110, 'cancelled': 1199})
check('no missing order amounts', orders.order_amount.notna().all())
completed = orders.status.eq('completed')
cancelled = orders.status.eq('cancelled')
check('no missing completed delivery times', orders.loc[completed, 'delivery_minutes'].notna().all())
check('all cancelled delivery times missing', orders.loc[cancelled, 'delivery_minutes'].isna().all())
check('completed revenue total', round(orders.loc[completed, 'order_amount'].sum(), 2) == 2998668.80)
check('40 products retained', len(products) == 40)

with sqlite3.connect(ROOT / 'data' / 'instamart.db') as con:
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    check('three SQLite tables exist', {'customers', 'products', 'orders'}.issubset(tables))
    check('SQLite order count', con.execute('SELECT COUNT(*) FROM orders').fetchone()[0] == 12309)

print('ALL CHECKPOINTS PASSED')
