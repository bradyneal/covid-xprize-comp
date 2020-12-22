import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm

ID_COLS = ['CountryName',
           'RegionName',
           'GeoID',
           'Date']
CASES_COL = ['NewCases']
NPI_COLS = ['C1_School closing',
            'C2_Workplace closing',
            'C3_Cancel public events',
            'C4_Restrictions on gatherings',
            'C5_Close public transport',
            'C6_Stay at home requirements',
            'C7_Restrictions on internal movement',
            'C8_International travel controls',
            'H1_Public information campaigns',
            'H2_Testing policy',
            'H3_Contact tracing',
            'H6_Facial Coverings']

def add_geoID(ds):
    
    if 'RegionName' in ds.columns:
        ds["GeoID"] = np.where(ds["RegionName"].isnull(),
                           ds["CountryName"] + "__nan",
                           ds["CountryName"] + ' / ' + ds["RegionName"])
    else:
        zone = ds['CountryName']
        zone = zone.str.split("/", expand=True)
        ds['GeoID'] = np.where(zone["RegionName"].isnull(),
                       zone["CountryName"],
                       zone["CountryName"] + ' / ' + zone["RegionName"])
    return ds


def select_metric(gdp_df, metric, metrics):
    metrics.remove(metric)
    gdp_df = gdp_df.drop(metrics,axis=1)
    
    return gdp_df


def load_quarterly_imf(path):
    
    df = pd.read_excel(path,
                       header=[1],
                       parse_dates=['Unnamed: 1'])

    df = df.rename({'Unnamed: 0': 'CountryName',
                'Unnamed: 1': 'Date'
                }, axis=1)

    qs = df['Date'].copy().str.replace(r'(Q\d) (\d+)', r'\2-\1')
    df.loc[:,'Date'] = pd.PeriodIndex(qs.values, freq='Q').to_timestamp() + pd.offsets.QuarterEnd(0)

    return df


def prep_gdp(gdp):
    
    col_name = 'National Accounts, Expenditure, Gross Domestic Product'\
    + ', Real, Percent Change, Previous Period, Percent'
    
    gdp = select_metric(gdp,
                        col_name,
                        list(gdp.columns[2:]))

    gdp = gdp.rename({gdp.columns[-1]: 'GDP growth'}, axis=1)
    
    gdp_uptodate = gdp[gdp['Date'] == '2020-06-30'].copy().dropna()

    gdp = gdp[gdp.CountryName.isin(set(gdp_uptodate.CountryName))]
    
    gdp.set_index('Date', inplace=True)

    return add_freq(gdp, 'Q')


def prep_ir(ir):
    
    col_name= 'Monetary and Financial Accounts, Interest Rates, Central Bank '\
    + 'Policy Rates, Monetary Policy-Related Interest Rate, Percent per Annum'
    ir = ir.rename({col_name: 'Interest Rate'}, axis=1)
    
    ir = ir.drop(ir.columns[-3:], axis=1)
    
    ir_uptodate = ir[ir['Date'] == '2020-06-30'].copy().dropna()
    
    ir = ir[ir.CountryName.isin(set(ir_uptodate.CountryName))]

    ir.set_index('Date', inplace=True)
    return ir

def prep_quarterly_npis(npis, empty_start_date):
    
    npis.set_index(['GeoID','Date'], inplace=True)
    
    npis = npis.groupby([pd.Grouper(level='GeoID'),
                 pd.Grouper(level='Date', freq='Q')]).mean()

    empty_dates = pd.date_range(empty_start_date, npis.index.get_level_values('Date').min() - pd.Timedelta(days=1) , freq='Q')

    npi_empty = pd.DataFrame(index = pd.MultiIndex.from_product([set(npis.index.get_level_values('GeoID')),
                                                                 empty_dates],
                                                                 names = ['GeoID', 'Date']))
    
    npis = npis.append(npi_empty).sort_values(by = ['GeoID','Date'])

    npis.reset_index(level='GeoID', inplace=True)
    npis.fillna(0, inplace=True)
    return add_freq(npis, 'Q')


def prep_econ_context(path='econ_data/84d3aea2-dc90-4186-8c0c-7e9a35aff981_Data.csv'):

    econ_context = pd.read_csv(path)
    econ_context = econ_context.drop(columns = ['Country Code',
                                          'Series Code'])
    econ_context = econ_context.replace('..', np.nan)

    econ_context.dropna(how='all').reset_index(inplace=True)
    econ_context = econ_context.dropna(axis=0, how='all', subset = ['2010 [YR2010]', '2011 [YR2011]',
           '2012 [YR2012]', '2013 [YR2013]', '2014 [YR2014]', '2015 [YR2015]',
           '2016 [YR2016]', '2017 [YR2017]', '2018 [YR2018]', '2019 [YR2019]',
           '2020 [YR2020]'])
    
    return econ_context


def get_empl_context(econ_context):
    econ_context_cols = set(econ_context['Series Name'])

    empl_cols = econ_context_cols.difference(['GDP (current US$)',
     'GDP, PPP (current international $)',
     'Gini index (World Bank estimate)',
     'Increase in poverty gap at $1.90 ($ 2011 PPP) poverty line due to out-of-pocket health care expenditure (% of poverty line)',
     'Population density (people per sq. km of land area)',
     'Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population)',
     'Poverty headcount ratio at $3.20 a day (2011 PPP) (% of population)',
     'Poverty headcount ratio at $5.50 a day (2011 PPP) (% of population)']
    )

    empl_context = econ_context[econ_context['Series Name'].isin(empl_cols)]
    
    empl_context = empl_context.melt(
        id_vars = ['Country Name', 'Series Name'])
    empl_context = empl_context.pivot(
        index=['Country Name', 'variable'], 
        columns = 'Series Name', 
        values='value')
    
    empl_context = empl_context.reset_index()

    empl_context.index = pd.to_datetime(empl_context['variable'].str.split(' ', expand=True)[0].values) + pd.offsets.YearEnd(0)
    
    empl_context.index.rename('Date', inplace=True)
    empl_context.columns.name = None
    empl_context.drop('variable', axis=1, inplace=True)

    empl_context.columns.name = None
    empl_context = empl_context.fillna(method='ffill')
    empl_context= empl_context.reset_index().set_index(['Date','Country Name'])
    empl_context = empl_context.astype('float32')

    empl_context = empl_context.unstack(level=[1]).resample('Q').asfreq(
        ).interpolate(how='linear').stack(level=1).reset_index(
        level='Country Name').sort_values(['Country Name', 'Date'])
    empl_context.rename({'Country Name':'CountryName'},axis=1)

    return empl_context



# def get_recent(empl_context):
    
#     dct = {country:{} for country in set(empl_context['Country Name'])}
    
#     for i in empl_context.index:
#         row = empl_context.loc[i]
#         last_valid = row[row['2010 [YR2010]':].last_valid_index()]
#         country = row['Country Name']
#         series = row['Series Name']
#         dct[country][series] = last_valid
    
#     df = pd.DataFrame.from_dict(dct, orient='index').dropna(
#            ).reset_index().rename({'index':'CountryName'}, axis=1)
    
#     df = df.set_index('CountryName')
#     return df


def get_endog(country, econ_data):
    return econ_data[econ_data.CountryName == country]['GDP growth']


def get_exog(country, econ_data):
    return sm.add_constant(econ_data[econ_data.CountryName == country].loc[:, 'C1_School closing_0.0':], has_constant='add')


def rename_imf_countries(gdp):

    country_names= {'China, P.R.: Hong Kong': None,
                    'Croatia, Rep. of': 'Croatia',
                    'Cyprus': None,
                    'Czech Rep.': 'Czech Republic',
                    'Estonia, Rep. of': 'Estonia',
                    'Euro Area': None,
                    'Latvia': None,
                    'Lithuania': None,
                    'Malta': None,
                    'Netherlands, The': 'Netherlands',
                    'Poland, Rep. of': 'Poland',
                    'Serbia, Rep. of': 'Serbia',
                    'Slovenia, Rep. of': 'Slovenia'}
    
    gdp = gdp.replace(country_names)

    return gdp


def add_freq(idx, freq=None):
    """Add a frequency attribute to idx, through inference or directly.

    Returns a copy.  If `freq` is None, it is inferred.
    """

    idx = idx.copy()
    if freq is None:
        if idx.freq is None:
            freq = pd.infer_freq(idx)
        else:
            return idx
    idx.freq = pd.tseries.frequencies.to_offset(freq)
    if idx.freq is None:
        raise AttributeError('no discernible frequency found to `idx`.  Specify'
                             ' a frequency string with `freq`.')
    return idx
