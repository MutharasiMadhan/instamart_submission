from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent
sql_text = (ROOT / 'sql' / 'queries.sql').read_text(encoding='utf-8')
# Split on the explicit Q labels; each block has exactly one SELECT statement.
blocks = []
current = []
for line in sql_text.splitlines():
    if line.strip().startswith('-- Q'):
        if current:
            statement = '\n'.join(current).strip()
            if statement:
                blocks.append(statement)
        current = [line]
    else:
        current.append(line)
if current:
    statement = '\n'.join(current).strip()
    if statement:
        blocks.append(statement)

with sqlite3.connect(ROOT / 'data' / 'instamart.db') as con:
    for i, block in enumerate(blocks, 1):
        statement = '\n'.join(line for line in block.splitlines() if not line.lstrip().startswith('--')).strip()
        if not statement:
            continue
        df = pd.read_sql_query(statement, con)
        print(f'\n=== Q{i} ({len(df)} rows) ===')
        print(df.to_string(index=False, max_rows=20))
        if i == 2:
            print('Q2 grand total INR {:.2f}'.format(df['revenue'].sum()))
        if i == 6:
            print('RANK differs from ROW_NUMBER:', int((df['spend_rank'] != df['spend_row_number']).sum()))
        if i == 7 and len(df):
            print('Last cumulative revenue INR {:.2f}'.format(df.iloc[-1]['cumulative_revenue']))
