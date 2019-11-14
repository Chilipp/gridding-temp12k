import pyleogrid as pg
import numpy as np
import pygam
import datetime as dt
import pandas as pd
import xarray as xr
from sklearn.neighbors import BallTree
import distributed


def is_between(da, left, right):
    return (da >= left) & (da <= right)


def get_ensemble_overlap(ens1, ens2):
    return (is_between(ens1, ens2.min(), ens2.max()),
            is_between(ens2, ens1.min(), ens1.max()))


def align_ensembles(ds, min_overlap=100):
    ds['aligned'] = xr.zeros_like(ds.age, dtype=bool)
    ds['alignment_base'] = xr.zeros_like(ds.age, dtype=bool)
    ds['unaligned_temperature_ensemble'] = ds['temperature_ensemble'].copy()
    groups = dict(ds.groupby(ds.TSid))
    max_id = max(groups, key=lambda ts_id: (groups[ts_id].age_ensemble.max() -
                                            groups[ts_id].age_ensemble.min()))
    aligned = groups.pop(max_id)
    aligned['alignment_base'][:] = True
    found_overlap = True

    agedim = ds.age.dims[0]

    while groups and found_overlap:
        found_overlap = False
        for ts_id in groups:
            m1, m2 = get_ensemble_overlap(aligned.age_ensemble,
                                          groups[ts_id].age_ensemble)
            if m1.sum() > min_overlap and m2.sum() > min_overlap:
                found_overlap = True
                ts_ds = groups.pop(ts_id)
                diff = (aligned.temperature_ensemble.values[m1.values].mean() -
                        ts_ds.temperature_ensemble.values[m2.values].mean())
                ts_ds['temperature_ensemble'] = (
                    ts_ds['temperature_ensemble'] + diff)
                aligned = xr.concat([aligned, ts_ds], dim=agedim)
                break

    aligned['aligned'][:] = True
    if groups:
        aligned = xr.concat([aligned] + list(groups.values()), dim=agedim)
    return aligned


class GAMEnsemble(pg.Ensemble):
    """A :class:`pyleogrid.Ensemble` using pseudo-gridding with GAM"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # use psyplot base to keep coords of unstructured datasets
        if self.target is not None:
            self.output_ds = self.target.psy.base

    def find_gridcells(self, target):
        conf = self.config

        full_coords = self.input_data.reset_coords()[
            [conf.lon, conf.lat]].to_dataframe()
        coords = full_coords.drop_duplicates().copy()

        target_lon = target.psy.decoder.get_x(target.psy[0])
        target_lat = target.psy.decoder.get_y(target.psy[0])

        # load grid cell centers
        tree = BallTree(np.r_[target_lon.values[None],
                              target_lat.values[None]].T)
        indices = tree.query(coords, return_distance=False)[:, 0]

        lonname = target_lon.name
        if lonname == conf.lon:
            lonname = conf.lon + "_grid"
        latname = target_lat.name
        if latname == conf.lat:
            latname = conf.lat + "_grid"

        coords[lonname] = target_lon.values[indices]
        coords[latname] = target_lat.values[indices]

        full_coords = full_coords.merge(coords, on=[conf.lon, conf.lat],
                                        how='left')
        agedim = self.input_data[conf.age].dims[0]
        self.input_data[lonname] = (agedim, full_coords[lonname])
        self.input_data[latname] = (agedim, full_coords[latname])

        self.input_data = self.input_data.set_coords(
            [lonname, latname]).sortby(
                [latname, lonname, conf.ds_id, conf.age])

        return self.input_data[lonname], self.input_data[latname]

    def predict(
            self, age=None, climate=None, target=None,
            size=None, client=None, quantiles=None, return_gam=False,
            return_time=False,
            **kwargs):

        conf = self.config

        if target is None:
            target = self.target

        if target is None:
            raise ValueError("No target elevation data supplied!")

        clon, clat = self.find_gridcells(target)

        if age is None:
            age = self.input_data.get(conf.age + '_ensemble',
                                      self.input_data[conf.age])
        if climate is None:
            climate = self.input_data.get(conf.climate + '_ensemble',
                                          self.input_data.get(conf.climate))
            if climate is None:
                raise ValueError("No climate data found!")

        data = self.input_data

        data[conf.climate + '_ensemble'] = climate
        data[conf.age + '_ensemble'] = age

        output = target.psy.base.copy()

        time = target.dims[0]  # time dimension
        space = target.dims[1]  # spatial dimension
        nt = output.dims[time]
        ns = output.dims[space]

        encoding = dict(coordinates=f'{clon.name} {clat.name}')

        output[conf.climate] = target.copy(
            data=np.full(target.shape, np.nan))
        if quantiles is not None:
            output[conf.climate + '_quantiles'] = xr.Variable(
                (time, space, 'quantile'),
                np.full((nt, ns, len(quantiles)), np.nan),
                encoding=encoding)
            output['quantile'] = ('quantile', quantiles)
        if size is not None:
            output[conf.climate + '_samples'] = xr.Variable(
                (time, conf.ensemble, space), np.full((nt, size, ns), np.nan),
                encoding=encoding)
        if return_gam:
            output[conf.climate + '_model'] = xr.Variable(
                space, np.full(ns, object, dtype=object),
                encoding=encoding)
        if return_time:
            output['time_needed'] = xr.Variable(
                space, np.zeros(ns, int), encoding=encoding)

        time = target.coords[target.dims[0]].values

        kwargs.update(dict(time=time, quantiles=quantiles, size=size,
                           return_gam=return_gam))

        target_idx = pd.MultiIndex.from_arrays(
            [target.psy.decoder.get_x(target.psy[0]).values,
             target.psy.decoder.get_y(target.psy[0]).values])

        kwargs['ensemble'] = self

        coords = pd.MultiIndex.from_arrays(
            [clon.values, clat.values])

        def iterate():
            locs = map(coords.get_loc, coords.drop_duplicates())
            if client is not None:
                kwargs['ensemble'] = client.scatter([self], broadcast=True)[0]
                futures = client.map(self._predict_cell_static, list(locs),
                                     **kwargs)
                for future, result in distributed.as_completed(
                        futures, with_results=True):
                    futures.remove(future)
                    yield result
            else:
                for sl in locs:
                    yield self._predict_cell_static(sl, **kwargs)

        for ret in pg.utils.log_progress(
                iterate(), len(coords.drop_duplicates())):
            key = ret[0]
            i = target_idx.get_loc(coords[key][0])
            try:
                i = i.start
            except AttributeError:
                pass
            output[conf.climate][:, i] = ret[1]
            j = 2
            if quantiles is not None:
                output[conf.climate + '_quantiles'][:, i, :] = ret[j]
                j += 1
            if size is not None:
                output[conf.climate + '_samples'][:, :, i] = ret[j]
                j += 1
            if return_gam:
                output[conf.climate + '_model'][i] = ret[j]
                j += 1
            if return_time:
                output['time_needed'][i] = ret[j]

        return output

    def _predict_cell(self, locs, time, quantiles=None, size=None,
                      align=True, anomalies=True, min_overlap=100,
                      return_gam=False, max_time_diff=200, return_time=False):

        t0 = dt.datetime.now()

        conf = self.config

        agedim = self.input_data[conf.age].dims[0]

        ds = self.input_data.isel(**{agedim: locs})

        climate = conf.climate + '_ensemble'
        age = conf.age + '_ensemble'
        modern = 'modern'

        ens_size = ds[age].shape[0]
        if align:
            ds = self.align_ensembles(ds, min_overlap=min_overlap)
        else:
            ds['aligned'] = (agedim, np.zeros(ds.dims[agedim], bool))

        if anomalies:
            # remove the anomaly for the aligned data
            aligned_ids = np.where(ds.aligned)[0]
            aligned = ds.isel(age=aligned_ids)
            # define modern via worldclim reference: 1970-2000
            is_modern = is_between(aligned[age], -50, -20).values
            if is_modern.sum() > 100:
                mean = aligned[climate].values[is_modern].mean()
            elif aligned.datum[aligned.alignment_base][0] == 'anom':
                mean = 0
            else:
                mean = aligned[modern][aligned.alignment_base][0].values
            ds[climate][:, aligned_ids] -= mean

            # remove anomaly for non-aligned data
            nonaligned = ds[conf.ds_id].where(~ds.aligned, drop=True)

            for ts_id in np.unique(nonaligned):
                ids = np.where(ds[conf.ds_id] == ts_id)[0]
                sub = ds.isel(**{agedim: ids})
                is_modern = is_between(sub.age_ensemble, -50, -20).values
                if is_modern.sum() >= 100:
                    mean = sub[climate].values[is_modern].mean()
                elif sub.datum[sub.alignment_base][0] == 'anom':
                    mean = 0
                else:
                    mean = sub[modern].values[np.newaxis]
                ds[climate][:, ids] -= mean

            # insert 0s for every timeseries in the ensemble for the reference
            # period at -35 BP (1985)
            zeros = np.zeros(np.unique(ds[conf.ds_id]).size * ens_size)
            x = np.r_[ds[age].values.ravel(), zeros - 35]
            y = np.r_[ds[climate].values.ravel(), zeros]

        gam = pygam.LinearGAM(pygam.s(0)).gridsearch(
            x[:, np.newaxis], y, progress=False)

        time = np.asarray(time)

        ret = (gam.predict(time), )
        modern = gam.predict(-35)
        if quantiles is not None:
            ret = ret + (gam.prediction_intervals(time, quantiles=quantiles), )
        if size is not None:
            ret = ret + (gam.sample(
                x[:, np.newaxis], y, sample_at_X=time, n_draws=size).T, )

        if anomalies:
            ret = tuple(arr - modern for arr in ret)

        # look how many samples in the ensemble fall into the `max_time_diff`
        # time interval around the predicted time
        tree = BallTree(ds[age].values.ravel()[:, np.newaxis])
        counts = tree.query_radius(time[:, np.newaxis], max_time_diff,
                                   count_only=True)

        idx = counts < 100
        if idx.any():
            for arr in ret:
                arr[idx] = np.nan
        if return_gam:
            ret = ret + (gam, )

        if return_time:
            ret = ret + ((dt.datetime.now() - t0).total_seconds(), )

        return ret

    def align_ensembles(self, ds, min_overlap=100):

        conf = self.config

        climate = conf.climate + '_ensemble'

        ds['aligned'] = xr.zeros_like(ds.age, dtype=bool)
        ds['alignment_base'] = xr.zeros_like(ds.age, dtype=bool)
        ds['unaligned_' + climate] = ds[climate].copy()
        groups = dict(ds.groupby(ds.TSid))
        max_id = max(groups, key=lambda ts_id: (
            groups[ts_id].age_ensemble.max() -
            groups[ts_id].age_ensemble.min()))
        aligned = groups.pop(max_id)
        aligned['alignment_base'][:] = True
        found_overlap = True

        agedim = ds.age.dims[0]

        while groups and found_overlap:
            found_overlap = False
            for ts_id in groups:
                m1, m2 = get_ensemble_overlap(aligned.age_ensemble,
                                              groups[ts_id].age_ensemble)
                if m1.sum() > min_overlap and m2.sum() > min_overlap:
                    found_overlap = True
                    ts_ds = groups.pop(ts_id)
                    diff = (
                        aligned[climate].values[m1.values].mean() -
                        ts_ds[climate].values[m2.values].mean())
                    ts_ds[climate] = ts_ds[climate] + diff
                    aligned = xr.concat([aligned, ts_ds], dim=agedim)
                    break

        aligned['aligned'][:] = True
        if groups:
            aligned = xr.concat([aligned] + list(groups.values()), dim=agedim)
        return aligned

    @staticmethod
    def _predict_cell_static(sl, **kwargs):
        ensemble = kwargs.pop('ensemble')
        return (sl, ) + ensemble._predict_cell(sl, **kwargs)


def test_align_ensembles():

    longest = pd.DataFrame({
        'TSid': [1] * 6,
        'age': np.arange(6),
        'temperature': np.arange(6)})

    overlapping = pd.DataFrame({
        'TSid': [2] * 3,
        'age': np.arange(2, 5) + 0.5,
        'temperature': np.arange(2, 5) - 2})

    not_overlapping = pd.DataFrame({
        'TSid': [3] * 3,
        'age': [7, 8, 9],
        'temperature': [-1] * 3})

    combined = pd.concat([longest, overlapping, not_overlapping],
                         ignore_index=True)

    combined['age_unc'] = 0.1

    ensemble = pg.Ensemble(combined.set_index('age'), ds_id='TSid',
                           climate='temperature')
    age_ensemble = ensemble.sample_ages(size=1000)
    ensemble.input_data['temperature_ensemble'] = age_ensemble.copy(
        data=np.tile(combined.temperature.values[None], (1000, 1)))

    aligned = align_ensembles(ensemble.input_data)
    assert aligned.aligned.where(aligned.TSid == 1, drop=True).all()
    assert aligned.aligned.where(aligned.TSid == 2, drop=True).all()
    assert not aligned.aligned.where(aligned.TSid == 3, drop=True).any()
    shifted_ds = aligned.where(aligned.TSid == 2, drop=True)
    assert np.allclose(
        shifted_ds.temperature_ensemble.values,
        shifted_ds.unaligned_temperature_ensemble.values + 2.5,
        atol=0.2)
