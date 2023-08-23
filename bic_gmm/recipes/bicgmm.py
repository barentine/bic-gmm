from PYME.recipes.base import register_module, ModuleBase
from PYME.recipes.traits import Input, Output, Enum, CStr, Int
import numpy as np
from PYME.IO import tabular
import logging


logger = logging.getLogger(__name__)

@register_module('GridBICGMM')
class GridBICGMM(ModuleBase):
    """Fit a Gaussian Mixture to a pointcloud, predicting component membership
    for each input point.

    Parameters
    ----------
    input_points: PYME.IO.tabular
        points to fit. Currently hardcoded to use x, y, and z keys.
    n: Int
        number of Gaussian components in the model for optimization mode `n` 
        and `bayesian`, or maxinum number of components for `bic`
    covariance: Enum
        type of covariance to use in the model
    label_key: str
        name of membership/label key in output datasource, 'gmm_label' by
        default
    output_labeled: PYME.IO.tabular
        input source with additional column indicating predicted component
        membership of each point
    
    Notes
    -----
    Directly implements or closely wraps scikit-learn mixture.GaussianMixture
    and mixture.BayesianGaussianMixture. See sklearn documentation for more 
    information.
    """
    input_points = Input('input')
    n = Int(1)
    # mode = Enum(('n', 'bic', 'gridsearch_bic', 'bayesian'))
    covariance = Enum(('full', 'tied', 'diag', 'spherical'))
    max_iter = Int(100)
    n_initializations = Int(1)
    n_grid_procs = Int(5)
    init_params = Enum(('kmeans', 'random'))
    label_key = CStr('gmm_label')
    output_labeled = Output('labeled_points')
    output_gmm = Output('gmm')
    
    def execute(self, namespace):
        from sklearn.mixture import GaussianMixture
        from PYME.IO import MetaDataHandler

        points = namespace[self.input_points]
        if np.all(points['z'] == points['z'][0]):
            # we have 2D data, make this faster for us
            logger.debug('Z is flat, using 2D GMM')
            X = np.stack([points['x'].astype(np.float32), points['y'].astype(np.float32)], axis=1)
        else:
            X = np.stack([points['x'].astype(np.float32), points['y'].astype(np.float32), points['z'].astype(np.float32)], axis=1)
        
        # n is treated as max
        best = self._check_bic_grid(X, 1, self.n, max_grid_points=self.n_grid_procs)
        print('BEST: %d' % best)
        gmm = GaussianMixture(n_components=best,
                                covariance_type=self.covariance,
                                max_iter=self.max_iter,
                                init_params=self.init_params,
                                n_init=self.n_initializations)
        predictions = gmm.fit_predict(X) + 1  # PYME labeling scheme
        log_prob = gmm.score_samples(X)
        if not gmm.converged_:
            logger.error('GMM fitting did not converge')
            predictions = np.zeros(len(points), int)
            log_prob = - np.inf * np.ones(len(points))
        
        out = tabular.MappingFilter(points)
        try:
            out.mdh = MetaDataHandler.DictMDHandler(points.mdh)
        except AttributeError:
            pass

        out.addColumn(self.label_key, predictions)
        out.addColumn(self.label_key + '_log_prob', log_prob)
        avg_log_prob = np.empty_like(log_prob)
        for label in np.unique(predictions):
            mask = label == predictions
            avg_log_prob[mask] = np.mean(log_prob[mask])
        out.addColumn(self.label_key + '_avg_log_prob', avg_log_prob)
        namespace[self.output_labeled] = out
        namespace[self.output_gmm] = gmm

    def _check_bic_grid(self, X, min_search, max_search, max_grid_points=5):
        import multiprocessing
        n_components = np.arange(min_search, max_search + 1, 
                                 int((max_search - min_search) / (max_grid_points - 1)), dtype=int)
        logger.debug('checking n_components: %s' % n_components)
        bic = np.zeros(len(n_components))
        params = [{'n_components': n_components[ind], 'covariance_type': self.covariance,
                   'max_iter': self.max_iter, 'init_params': self.init_params,
                   'n_init': self.n_initializations} for ind in range(len(n_components))]
        processes = []
        queue = multiprocessing.Queue()
        for ind in range(len(n_components)):
            p = multiprocessing.Process(target=_gmm, args=(X, params[ind], queue))
            p.start()
            processes.append(p)
        results = []
        for p in processes:
            p.join()
            results.append(queue.get())
        
        logger.debug(results)
        results = sorted(results, key=lambda x: x[0])
        bic = np.asarray([results[ind][1] for ind in range(len(results))])
            
        best_ind = np.argmin(bic)
        best = n_components[best_ind]
        logger.debug('Best BIC: %d' % best)
        min_search = n_components[max(best_ind - 1, 0)] + 1
        max_search = n_components[min(best_ind + 1, len(n_components) - 1)] - 1
        # check if we finished
        if best_ind == 0 or best_ind == len(n_components) - 1:
            # we're done - on a rail, just catching this to avoid the next elif
            logger.debug('Finished - on a rail')
        elif n_components[1] - n_components[0] > 1:
            logger.debug('current minimum with %d components' % best)
            logger.debug('homing search from %d to %d' % (min_search, max_search))
            next_n = min(max_search - min_search, max_grid_points)
            best = self._check_bic_grid(X, min_search, max_search, next_n)

        logger.debug('Finished BIC search, best: %d' % best)
        # we're done
        return best

def _gmm(X, gmm_args, queue):
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(**gmm_args)
    n = gmm_args['n_components']
    gmm.fit(X)
    bic = gmm.bic(X)
    logger.debug('%d BIC: %f' % (n, bic))
    queue.put((n, float(bic)))


@register_module('PredictLabelsFromGMM')
class PredictLabelsFromGMM(ModuleBase):
    input_points = Input('input')
    input_gmm = Input('gmm')
    label_key = CStr('gmm_label')
    output_points = Output('gmm_labeled')

    def run(self, input_points, input_gmm):
        """

        Parameters
        ----------
        input_points : PYME.IO.tabular.TabularBase
            _description_
        input_gmm : klearn.mixture.GaussianMixture
            _description_

        Returns
        -------
        _type_
            _description_
        """
        from PYME.IO.tabular import MappingFilter
        if np.all(input_points['z'] == input_points['z'][0]):
            # we have 2D data, make this faster for us
            logger.debug('Z is flat, using 2D GMM')
            X = np.stack([input_points['x'].astype(np.float32), input_points['y'].astype(np.float32)], axis=1)
        else:
            X = np.stack([input_points['x'].astype(np.float32), 
                          input_points['y'].astype(np.float32), 
                          input_points['z'].astype(np.float32)], axis=1)
        
        predictions = input_gmm.predict(X) + 1  # PYME labeling scheme
        log_prob = input_gmm.score_samples(X)

        output_points = MappingFilter(input_points)
        try:
            output_points.mdh = input_points.mdh
        except AttributeError:
            pass
        
        output_points.addColumn(self.label_key, predictions)
        output_points.addColumn(self.label_key + '_log_prob', log_prob)
        avg_log_prob = np.empty_like(log_prob)
        for label in np.unique(predictions):
            mask = label == predictions
            avg_log_prob[mask] = np.mean(log_prob[mask])
        output_points.addColumn(self.label_key + '_avg_log_prob', avg_log_prob)
        return output_points


@register_module('AddMeasurementsByLabel')
class AddMeasurementsByLabel(ModuleBase):
    input_points = Input('input')
    input_measurements = Input('clusterMeasures')
    label_key = CStr('label')
    output_points = Output('annotated_points')

    def run(self, input_points, input_measurements):
        """
        Propagate measurements from e.g. MeasureClusters3D back to points they were calcualted from.
        This is particularly useful for visualizing e.g. the gyration radius of a cluster in PYMEVis.

        Parameters
        ----------
        input_points : PYME.IO.tabular.TabularBase
            points used to generate the measurements
        input_measurements : PYME.IO.tabular.TabularBase
            measurements to propagate back to the points

        Returns
        -------
        PYME.IO.tabular.TabularBase
            point data with new columns added, one for each scalar measurement in input_measurements, which
            can be accessed by '<label_key>_<measurement_key>', e.g. 'clumpIndex_gyrationRadius'.
        
        """
        from PYME.IO.tabular import MappingFilter
        
        # only propagate 1D measurements
        annotations = dict()
        for k in input_measurements.keys():
            if input_measurements[k].ndim == 1:
                annotations[k] = np.zeros(len(input_points), dtype=input_measurements[k].dtype)
        
        try:
            labels = np.unique(input_measurements[self.label_key])
        except KeyError:
            logger.exception('Label key %s not found in input_measurements, RISKY: continuing with assumption measurements are sorted by label and all present' % self.label_key)
            labels = np.arange(1, len(input_measurements) + 1)  # MeasureClusters3D ignores the unclustered points 'label 0' so index 0 corresponds to 'label 1'
        
        for label in labels:
            points_mask = label == input_points[self.label_key]
            try:
                measurement_mask = label == input_measurements[self.label_key]
            except KeyError:
                measurement_mask = label - 1  # MeasureClusters3D ignores the unclustered points 'label 0' so index 0 corresponds to 'label 1'
            
            for k in annotations.keys():
                annotations[k][points_mask] = input_measurements[k][measurement_mask]

        output_points = MappingFilter(input_points)
        try:
            output_points.mdh = input_points.mdh
        except AttributeError:
            pass
        
        for k in annotations.keys():
            output_points.addColumn(self.label_key + '_' + k, annotations[k])
        
        return output_points
