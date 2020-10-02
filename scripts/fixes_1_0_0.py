"""LiPD Database fixes for version Temperature12K version 1.0.0"""
import numpy as np
import pandas as pd

def fix_invalid_series(df):
    to_reject = [
        'WEBa31332021',
        'WEBe2583d1f1',
        'WEB132596a41',
        'WEBd98cf2501',
        'WEBa31332022',
        'WEB294348af2',
        'WEB17b6eb432',
        'WEBfa1a1dfc2',
        'WEBa31332023',
        'WEBf03210593',
        'WEBee365cf33',
        'WEBf717905c3',
        'WEBa31332024',
        'WEBe514d1124',
        'WEBbd43db314',
        'WEB04980cd44',
        'WEBa31332025',
        'WEB68bd240f5',
        'WEBeda5d1445',
        'WEBb4d01ad65',
        'WEBa31332026',
        'WEB1471566f6',
        'WEBdbc993946',
        'WEB55c8fe9b6',
        'WEBa31332027',
        'WEB9ddc33ed7',
        'WEB7aee36bc7',
        'WEB185f2ee67',
        'WEB48ec3b148',
        'WEBbbb69cf38',
        'WEBdb5bec388',
        'WEBa31332029',
        'WEBe85bf21d9',
        'WEB0292cb449',
        'WEB12f682829',
        'WEBa31332028',
        'RRY0jogRrD7',
        'ROCezuKEjM5',
        'RoX4PAzwaJh',
        'RHN8ULGqRAs',
        'M2Lp7ovi4zuP',
        'M2Leb83o3ngP',
        'M2Ln0os2orwP',
        'M2L92hd3tfiP',
        'T2L_WIND28K_d18o_ruber_SST_from_d18o_ruber',
        'PYTW8HHSCX2',
        'PYT909ZJ7SN',
        'PYTCYF0YQTU',
        'M2Lqd7jeqa8P',
        'T2L_B997_321_d18o_pachyderma_SST_from_d18o_pachyderma',
        'T2L_B997_324_d18o_pachyderma_SST_from_d18o_pachyderma',
        'T2L_HU73_031_7_d18o_ruber_SST_from_d18o_ruber',
        'T2L_HU73_031_7_d18o_pachyderma_SST_from_d18o_pachyderma',
        'M2L6n22fvlcP',
        'M2L2jfcsqvbP',
        'M2L55sfpyavP',
        'T2L_OCE326_GGC26_d18o_pachyderma_SST_from_d18o_pachyderma',
        'T2L_GeoB5844_2_d18o_ruber_SST_from_d18o_ruber',
        'T2L_ERDC_092BX_d18o_sacculifer_SST_from_d18o_sacculifer',
        'T2L_RC24_08GC_mgca_ruber_SST_from_mgca_ruber',
        'M2Lwm503978P',
        'RpbgDFH9OsU',
        'T2L_JR51GC_35_uk37_SST_from_uk37',
        'T2L_MD97_2120_d18o_bulloides_SST_from_d18o_bulloides',
        'ReEnzeIOExA',
        ]
    df = df[~df.index.isin(to_reject)]

    # reject samples of FanLake.Foster.2016 and Yanou.Foster.2016 above 10.3°C
    to_reject2 = ["WEB6df77dbb", "WEB911b3b41"]
    df = df[~((df.index.isin(to_reject2)) & (df.temperature > 10.3))]

    return df

def fix_datum(df):
    df.loc[['WEBbc9e627b', 'Ry7X6BBn8Ko', 'RwTmo8nvdZo', 'R7xfSla4XGC'],
           'datum'] =  'anom'
    return df

def fix_MD02_2515(df):
    idx = df.index.get_loc('T2L_MD02_2515_tex86_SST_from_tex86')
    df.loc[idx, 'temperature'] += 8
    return df

def fix_A7(df):
    ts_id = 'T2L_A7_mgca_ruber_SST_from_mgca_ruber'
    df.loc[ts_id, 'seasonality'] = 'summer'
    df.loc[ts_id, 'seas'] = 'summer'
    df.loc[ts_id, 'pi'] = df.loc[ts_id, 'tavg_jja_pi'].values[0]
    df.loc[ts_id, 'worldclim'] = df.loc[ts_id, 'tavg_jja_wc'].values[0]


def fix_MV99_PC14(df):
    ts_id = 'T2L_MD95_2015_uk37_SST_from_uk37'
    df.loc[ts_id, 'seasonality'] = 'winter'
    df.loc[ts_id, 'seas'] = 'winter'
    df.loc[ts_id, 'pi'] = df.loc[ts_id, 'tavg_djf_pi'].values[0]
    df.loc[ts_id, 'worldclim'] = df.loc[ts_id, 'tavg_djf_wc'].values[0]


def fix_MV99_PC14(df):
    ts_id = 'T2L_MD95_2015_uk37_SST_from_uk37'
    df.loc[ts_id, 'seasonality'] = 'summerOnly'
    df.loc[ts_id, 'seas'] = 'summer'
    df.loc[ts_id, 'pi'] = df.loc[ts_id, 'tavg_jja_pi'].values[0]
    df.loc[ts_id, 'worldclim'] = df.loc[ts_id, 'tavg_jja_wc'].values[0]


def fix_Andy_Szeicz_1995(df):
    df.loc['PYTMLFNC0A7', 'seasonality'] = 'winter'
    df.loc['PYTCLN8P1CL', 'seasonality'] = 'summer'
    return recalculate(df, ['RSvtIc0N66Y'])


def fix_H214(df):
    return recalculate(df, ['RypmpVeFWPB'])


def fix_seasonality(df):
    df.loc[['PYTE08I3BN3', 'PYTC38VBYRL'], 'seasonality'] = 'summer'
    df.loc[['PYTHEC2VIP4', 'PYT7ZDYFS88'], 'seasonality'] = 'winter'
    return df


def fix_MD99_2251(df):
    ts_id = 'T2L_MD99_2251_mgca_bulloides_SST_from_mgca_bulloides'
    df.loc[ts_id, 'seasonality'] = 'summerOnly'
    df.loc[ts_id, 'seas'] = 'summer'
    df.loc[ts_id, 'pi'] = df.loc[ts_id, 'tavg_aug_pi'].values[0]
    df.loc[ts_id, 'worldclim'] = df.loc[ts_id, 'tavg_aug_wc'].values[0]
    return df


def fix_MD99_2256_Jennings_2015(df):
    ts_id = 'LPD6e872195'
    df.loc[ts_id, 'seasonality'] = 'summer'
    df.loc[ts_id, 'seas'] = 'summer'
    df.loc[ts_id, 'pi'] = df.loc[ts_id, 'tavg_jja_pi'].values[0]
    df.loc[ts_id, 'worldclim'] = df.loc[ts_id, 'tavg_jja_wc'].values[0]


def fix_VM28_122(df):
    idx = df.index.get_loc('T2L_VM28_122_mgca_ruber_SST_from_mgca_ruber')
    df.loc[idx, 'temperature'] += 2.45


def fix_TR163_22(df):
    idx = df.index.get_loc('T2L_TR163_22_mgca_ruber_SST_from_mgca_ruber')
    df.loc[idx, 'temperature'] += 3.3


def fix_modern_values(df):
    df.loc['T2L_VM12_107_mgca_ruber_SST_from_mgca_ruber',
           ['pi', 'worldclim']] = 25.6
    df.loc['LPD575ea390', ['pi', 'worldclim']] = 21.0
    df.loc['RchrJSmu2H9', ['pi', 'worldclim']] = 13.0
    df.loc['WEB4f8fa23d', ['pi', 'worldclim']] = 18.0
    df.loc['RWShShxDbGs_SST_from_Uk37', ['pi', 'worldclim']] = 5.75
    df.loc['WEB1fa182ff', ['pi', 'worldclim']] = 26.5
    df.loc['RPZj5YKrFr0', ['pi', 'worldclim']] = 15.0


def fix_single_seasons(df):
    """Check whether the summerOnly and winterOnly values are actually correct
    """
    ids = df.loc[df.seasonality.isin(
        ['summerOnly', 'winterOnly'])].index.drop_duplicates()
    for ts_id in ids:
        row = df.loc[ids].iloc[0]
        if len(df.loc[df.dataSetName == row.dataSetName,
                      'seasonality'].unique()) > 1:
            df.loc[ts_id, 'seasonality'] = df.loc[ts_id, 'seas'].values[0]


def recalculate(df, ids):

    if 'recalculated' not in df.columns:
        df['recalculated'] = False

    recalc = {}
    for ts_id in ids:
        row = df.loc[ts_id].iloc[0]
        others = df[df.dataSetName == row.dataSetName]
        others = others[~others.index.duplicated()]
        if len(others) < 3:
            continue
        summer = others[(others.seas == 'summer') &
                        (others.proxy == row.proxy)]
        winter = others[(others.seas == 'winter') &
                        (others.proxy == row.proxy)]
        if (len(summer) == 1 and len(winter) == 1 and
                summer.datum.values[0] == winter.datum.values[0]):
            recalc[ts_id] = np.r_[summer.index, winter.index]

    print('Dropping %i records' % len(df.loc[ids]))
    df.drop(ids, inplace=True)

    recalculated_data = []

    for target, ids in recalc.items():
        ids_data = df.loc[ids].reset_index().drop_duplicates(['age', 'TSid'])

        ids_ann = ids_data.pivot('age', 'seas', 'temperature').interpolate(
            method='linear', limit_area='inside').mean(axis=1).rename(
                'temperature').to_frame()
        ids_ann['pi'] = np.mean(ids_data['pi'].unique())
        ids_ann['worldclim'] = np.mean(ids_data['worldclim'].unique())
        ids_ann['TSid'] = target
        ids_ann['seasonality'] = 'annual'
        ids_ann['seas'] = 'annual'
        ids_ann['variableName'] = 'temperatureComposite2'
        # merge age uncertainties
        ids_ann = ids_ann.merge(
            ids_data.pivot('age', 'seas', 'age_unc').interpolate(
                method='linear').mean(axis=1).rename('age_unc'),
            on='age')
        # merge temperature uncertainties
        ids_ann = ids_ann.merge(
            ids_data.pivot('age', 'seas', 'temp_unc').interpolate(
                method='linear').mean(axis=1).rename('temp_unc'),
            on='age').reset_index()
        # merge the rest
        for col in [col for col in ids_data.columns if col not in ids_ann.columns]:
            ids_ann[col] = ids_data.iloc[0][col]
        ids_ann['recalculated'] = True
        recalculated_data.append(ids_ann)

    print("Appending %s records" % sum(map(len, recalculated_data)))
    df = pd.concat([df.reset_index()] + recalculated_data, ignore_index=True,
                   sort=False)

    df.set_index('TSid', inplace=True)

    return df



