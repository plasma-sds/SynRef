import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from math import inf
from scipy.optimize import least_squares

# --- PARAMETERS ---
tsv_file = '/home/attilas/Reflectometry/antenna_util/cleaned_temp.tsv'
out_csv = '/home/attilas/Reflectometry/antenna_util/fitted_traces.csv'
out_plot = '/home/attilas/Reflectometry/antenna_util/fit_diagnostics.png'
max_degree = 12            # maximum polynomial degree to consider
rcond_regular = 1e-12      # regularization for polyfit-like solve

# --- HELPERS ---
def read_tsv_auto(fname):
    # read TSV, skip commented lines starting with #
    raw = []
    with open(fname, 'r') as f:
        for line in f:
            if line.strip().startswith('#') or line.strip()=='':
                continue
            raw.append(line)
    # use pandas to read remaining lines as TSV with possible unequal columns
    from io import StringIO
    s = ''.join(raw)
    df = pd.read_csv(StringIO(s), sep='\t', header=[0,1], engine='python')
    return df

def extract_traces(df):
    # The TSV in your file has columns in pairs (Axis1, Axis2) for Trace2 and Trace4.
    # We'll flatten to numeric columns and drop all-NaN columns.
    # Convert all values to float, replace empty strings by NaN.
    df_flat = df.copy()
    # flatten multiindex columns if present
    if isinstance(df_flat.columns, pd.MultiIndex):
        df_flat.columns = ['_'.join([str(c).strip() for c in col if str(c).strip()]) for col in df_flat.columns.values]
    else:
        df_flat.columns = [str(c).strip() for c in df_flat.columns]
    # try convert
    for c in df_flat.columns:
        df_flat[c] = pd.to_numeric(df_flat[c].astype(str).str.replace('"','').str.strip(), errors='coerce')
    # drop empty columns
    df_flat = df_flat.dropna(axis=1, how='all')
    return df_flat

def poly_fit_regularized(x, y, deg, reg=0.0):
    # Build Vandermonde and solve (X^T X + reg I) a = X^T y
    V = np.vander(x, N=deg+1, increasing=False)  # highest first
    XtX = V.T @ V
    if reg>0:
        XtX = XtX + reg * np.eye(XtX.shape[0])
    Xty = V.T @ y
    coeffs = np.linalg.solve(XtX, Xty)
    return coeffs

def evaluate_poly(coeffs, x):
    # coeffs: highest-first (numpy style)
    return np.polyval(coeffs, x)

def select_degree_aicc(x, y, max_deg=10, reg=1e-12):
    n = x.size
    best = {'deg':None,'aicc':inf,'coeffs':None,'rss':None}
    for deg in range(1, max_deg+1):
        try:
            coeffs = poly_fit_regularized(x, y, deg, reg=reg)
        except np.linalg.LinAlgError:
            continue
        yhat = evaluate_poly(coeffs, x)
        rss = np.sum((y - yhat)**2)
        k = deg+1
        if rss <= 0:
            rss = 1e-20
        aic = n * np.log(rss / n) + 2*k
        aicc = aic + (2*k*(k+1)) / max(1, n - k - 1)
        if aicc < best['aicc']:
            best.update({'deg':deg, 'aicc':aicc, 'coeffs':coeffs, 'rss':rss})
    return best

# --- MAIN ---
df_raw = read_tsv_auto(tsv_file)
df = extract_traces(df_raw)

# Expect pairs: Axis1, Axis2 for each trace. We'll assume first column is x for trace1, second is y for trace1, etc.
cols = df.columns.tolist()
if len(cols) % 2 != 0:
    # some files have two traces but one longer - try to align by position pairs
    pass

traces = []
for i in range(0, len(cols), 2):
    try:
        xcol = cols[i]
        ycol = cols[i+1]
    except IndexError:
        break
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    # remove NaNs and ensure arrays
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if x.size==0:
        continue
    traces.append({'x':x, 'y':y, 'xcol':xcol, 'ycol':ycol})

# Fit each trace
results = []
plt.figure(figsize=(8,6))
colors = ['C0','C1','C2','C3']
for idx, t in enumerate(traces):
    x = t['x']
    y = t['y']
    # sort by x
    sidx = np.argsort(x)
    x = x[sidx]
    y = y[sidx]
    # choose degree using AICc up to max_degree
    best = select_degree_aicc(x, y, max_deg=min(max_degree, max(3, int(len(x)//5))), reg=1e-9)
    deg = best['deg']
    coeffs = best['coeffs']
    yhat = evaluate_poly(coeffs, x)
    rss = best['rss']
    results.append({'deg':deg, 'coeffs':coeffs, 'rss':rss, 'x':x, 'y':y, 'yhat':yhat,
                    'xcol':t['xcol'], 'ycol':t['ycol']})
    # plot
    plt.scatter(x, y, s=8, color=colors[idx%len(colors)], label=f'data {idx+1}')
    plt.plot(x, yhat, color=colors[idx%len(colors)], linewidth=2, label=f'poly deg{deg} fit {idx+1}')

coeffs_1 = results[0]['coeffs']
print(f'Selected polynomial coefficients (highest degree first): {coeffs_1}')
coeffs_2 = results[1]['coeffs']
print(f'Selected polynomial coefficients (highest degree first): {coeffs_2}')


plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(out_plot, dpi=200)
plt.show()

# Save outputs: CSV with columns x, y, yhat_trace1, yhat_trace2...
out_df = pd.DataFrame()
for i, r in enumerate(results):
    colx = f'x_{i+1}'
    coly = f'y_{i+1}'
    out_df[colx] = r['x']
    out_df[coly] = r['y']
    out_df[f'yhat_{i+1}'] = r['yhat']

out_df.to_csv(out_csv, index=False)

# Print coefficients in readable form
for i, r in enumerate(results):
    coeffs = r['coeffs']
    deg = r['deg']
    print(f'Trace {i+1}: selected degree = {deg}, RSS={r["rss"]:.3e}')
    print('coeffs (highest->lowest):')
    print(repr(coeffs))
    print()