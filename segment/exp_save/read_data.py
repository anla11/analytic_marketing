import pandas as pd

def prepare_fulldata(nrows = None):
    df = None
    if nrows is None:
        df = pd.read_excel("http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx", dtype = "str")
    else:
        df = pd.read_excel("http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx", dtype = "str", nrows=nrows)
    # 2. Preprocess Data
    df.dropna(inplace = True)
    df["Quantity"] = df["Quantity"].astype(int)
    df["UnitPrice"] = df["UnitPrice"].astype(float)
    df['InvoiceDate']= pd.to_datetime(df['InvoiceDate'])
    # Remove register without CustomerID
    df = df[~(df.CustomerID.isnull())]
    # Remove negative or return transactions
    df = df[~(df.Quantity<0)]
    df = df[df.UnitPrice>0]
    df["TotalPrice"] = df['UnitPrice']*df['Quantity']
    return df

def prepare_monthlydata(full_data, date_col, month_select='2010-12'): #default format: yyyy-mm-dd
    df_filtertime = full_data.copy()
    df_filtertime['Month'] = df_filtertime[date_col].apply(lambda r: str(r)[:7])
    df_filter = df_filtertime[df_filtertime['Month'] == month_select]
    return df_filter