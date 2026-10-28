from pathlib import Path
import pandas as pd

RAW = Path('data/raw')
for name in ['customers', 'products', 'orders', 'tickets']:
    path = RAW / f'{name}.csv'
    df = pd.read_csv(path, keep_default_na=False if name == 'tickets' else True)
    print(f'\n=== {name.upper()} ===')
    print('shape:', df.shape)
    print('\ndtypes:\n', df.dtypes.to_string())
    missing = df.isna().sum()
    # Empty strings are blanks for audit purposes; do not convert ticket text blanks to NaN.
    blank = df.apply(lambda col: col.astype('string').str.strip().eq('').sum())
    counts = pd.DataFrame({'missing_count': missing, 'missing_pct': (missing / len(df) * 100).round(2),
                           'blank_string_count': blank})
    print('\nmissing/blank by column:\n', counts.to_string())

orders = pd.read_csv(RAW / 'orders.csv', keep_default_na=False)
customers = pd.read_csv(RAW / 'customers.csv', keep_default_na=False)
amount = pd.to_numeric(orders['order_amount'], errors='coerce')
delivery = pd.to_numeric(orders['delivery_minutes'], errors='coerce')
ages = pd.to_numeric(customers['age'].replace('', pd.NA), errors='coerce')
summary = {
    'raw orders rows': len(orders),
    'exact duplicate rows': int(orders.duplicated().sum()),
    'pending rows': int(orders.status.eq('pending').sum()),
    'distinct order_id': int(orders.order_id.nunique()),
    'non-numeric order_amount': int(amount.isna().sum()),
    'negative delivery_minutes': int((delivery < 0).sum()),
    'blank delivery_minutes': int(orders.delivery_minutes.astype(str).str.strip().eq('').sum()),
    'blank age': int(customers.age.astype(str).str.strip().eq('').sum()),
    'age outside 10-100': int(((ages < 10) | (ages > 100)).sum()),
    'distinct raw city spellings': int(customers.city.nunique()),
    'distinct non-blank raw gender spellings': int(customers.loc[customers.gender.astype(str).str.strip().ne(''), 'gender'].nunique()),
    'blank gender': int(customers.gender.astype(str).str.strip().eq('').sum()),
}
print('\n=== REFERENCE AUDIT FIGURES ===')
print(pd.Series(summary).to_string())
