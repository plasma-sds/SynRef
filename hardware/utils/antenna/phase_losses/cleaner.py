import sys
from io import StringIO
import pandas as pd

fname = '/home/attilas/Reflectometry/antenna_util/balanis_F13.23.tsv'

# 1) Print the first ~80 raw lines for inspection
print('--- Raw file head (first 80 lines) ---')
with open(fname, 'r', encoding='utf-8', errors='replace') as f:
    for i, L in enumerate(f, 1):
        if i <= 80:
            print(f'{i:03d}: {L.rstrip()}')
        else:
            break
print('--- end head ---\n')

# 2) Read and produce a cleaned version: remove comment lines, trim trailing spaces,
#    replace occurrences of quote + tab or tab + quote to ensure consistent delimiting,
#    collapse double-double-quotes to single, remove stray quotes at line ends.
clean_lines = []
with open(fname, 'r', encoding='utf-8', errors='replace') as f:
    for i, L in enumerate(f, 1):
        s = L.rstrip('\n\r')
        if s.lstrip().startswith('#') or s.strip() == '':
            # skip comment/blank lines
            continue
        # Basic fixes:
        s = s.replace('""', '"')               # collapse double quotes
        s = s.replace('"\t', '\t')             # if a quote immediately precedes a tab, remove it
        s = s.replace('\t"', '\t')             # if a quote immediately follows a tab, remove it
        # if line starts or ends with stray quote, remove it
        if s.startswith('"') and not s.count('"') % 2 == 0:
            s = s.lstrip('"')
        if s.endswith('"') and not s.count('"') % 2 == 0:
            s = s.rstrip('"')
        clean_lines.append(s + '\n')

# write cleaned file for inspection
clean_fname = 'Reflectometry/antenna_util/cleaned_temp.tsv'
with open(clean_fname, 'w', encoding='utf-8') as cf:
    cf.writelines(clean_lines)
print(f'Wrote cleaned file: {clean_fname} ({len(clean_lines)} data lines)\n')

# 3) Try to parse cleaned file with pandas using python engine and no strict quoting
try:
    df = pd.read_csv(clean_fname, sep='\t', engine='python', header=[0,1], comment='#')
    print('Parsed with header=[0,1]. Columns:')
    print(df.columns.tolist())
except Exception as e:
    print('Parsing with header=[0,1] failed:', e)
    try:
        df = pd.read_csv(clean_fname, sep='\t', engine='python', header=None, comment='#', on_bad_lines='warn')
        print('Parsed with header=None. Shape:', df.shape)
        print('First rows:')
        print(df.head())
    except Exception as e2:
        print('Tolerant parse also failed:', e2)
        # fallback: manual split lines
        print('Attempting manual split on tabs (fallback).')
        rows = [line.rstrip('\n') .split('\t') for line in clean_lines]
        maxcols = max(len(r) for r in rows)
        import csv
        out_csv = 'manual_split.csv'
        with open(out_csv, 'w', newline='', encoding='utf-8') as out:
            writer = csv.writer(out)
            for r in rows:
                # pad short rows
                if len(r) < maxcols:
                    r = r + ['']*(maxcols - len(r))
                writer.writerow(r)
        print('Wrote manual-split CSV:', out_csv)
        print('You can inspect it with a spreadsheet editor.')

# If parse succeeded, show a summary
if 'df' in locals():
    print('\nDataFrame summary:')
    print(df.info())
    print(df.head(10))