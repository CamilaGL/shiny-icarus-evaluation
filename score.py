'''Evaluation script to get metrics of segmentations against ground truth labels.

Script partially derived from: 
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). 
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. 
Nature Methods, 1-9.

cldice implementation based on: 
Shit, S., Paetzold, J. C., Sekuboyina, A., Ezhov, I., Unger, A., Zhylka, A., ... & Menze, B. H. (2021). 
clDice-a novel topology-preserving loss function for tubular structure segmentation. 
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 16560-16569).

requirements:
    python
    numpy
    SimpleITK
    scikit-image
'''

import numpy as np
import os
import collections
import inspect
from multiprocessing.pool import Pool
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
from skimage.morphology import skeletonize

# ----------------------------------------------- metrics

def assert_shape(test, reference):

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)


class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)
        self.skirefcl = None
        self.skitestcl = None

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def set_skirefcl(self, refcl):
        self.skirefcl=refcl
        self.resetclski()
    
    def set_skitestcl(self, testcl):
        self.skitestcl=testcl
        self.resetclski()

    def resetclski(self):
        self.clp2vollintersectski = None
        self.clp2volltotalclski = None
        self.cll2volpintersectski = None
        self.cll2volptotalclski = None

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full

    #clDice
    def compute_clDice(self):
        if self.testcl is not None and self.referencecl is not None:
            self.clp2vollintersect = int(((self.testcl != 0) * (self.reference != 0)).sum())
            self.clp2volltotalcl = int((self.testcl!=0).sum())
            self.cll2volpintersect = int(((self.referencecl != 0) * (self.test != 0)).sum())
            self.cll2volptotalcl = int((self.referencecl!=0).sum())
        else:
            self.clp2vollintersect = 0
            self.clp2volltotalcl = 0
            self.cll2volpintersect = 0
            self.cll2volptotalcl = 0

    def get_clvalues(self):

        for entry in ( self.clp2vollintersect, self.clp2volltotalcl, self.cll2volpintersect, self.cll2volptotalcl):
            if entry is None:
                self.compute_clDice()
                break

        return self.clp2vollintersect, self.clp2volltotalcl, self.cll2volpintersect, self.cll2volptotalcl

    def compute_skiclDice(self):
        if self.skitestcl is not None and self.skirefcl is not None:
            self.clp2vollintersectski = int(((self.skitestcl != 0) * (self.reference != 0)).sum())
            self.clp2volltotalclski = int((self.skitestcl!=0).sum())
            self.cll2volpintersectski = int(((self.skirefcl != 0) * (self.test != 0)).sum())
            self.cll2volptotalclski = int((self.skirefcl!=0).sum())
        else:
            self.clp2vollintersectski = 0
            self.clp2volltotalclski = 0
            self.cll2volpintersectski = 0
            self.cll2volptotalclski = 0

    def get_skiclvalues(self):
        for entry in ( self.clp2vollintersectski, self.clp2volltotalclski, self.cll2volpintersectski, self.cll2volptotalclski):
            if entry is None:
                self.compute_skiclDice()
                break

        return self.clp2vollintersectski, self.clp2volltotalclski, self.cll2volpintersectski, self.cll2volptotalclski
 

def dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(2. * tp / (2 * tp + fp + fn))


def precision(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp))


def recall(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fn))

def skiClPrecision(test=None, reference=None, confusion_matrix=None, **kwargs):
    #with skimage skeletonize
    """cl precision"""
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    clp2vollintersect, clp2volltotalcl, cll2volpintersect, cll2volptotalcl = confusion_matrix.get_skiclvalues()

    clp2voll = float(clp2vollintersect/clp2volltotalcl)
    return clp2voll

def skiClRecall(test=None, reference=None, confusion_matrix=None, **kwargs):
    #with skimage skeletonize
    """cl recall"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    clp2vollintersect, clp2volltotalcl, cll2volpintersect, cll2volptotalcl = confusion_matrix.get_skiclvalues()

    cll2volp= float(cll2volpintersect/cll2volptotalcl)
    return cll2volp


def skiClDice(test=None, reference=None, confusion_matrix=None, **kwargs):
    #with skimage skeletonize
    """clDice"""
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    clp2vollintersect, clp2volltotalcl, cll2volpintersect, cll2volptotalcl = confusion_matrix.get_skiclvalues()
    #precision
    clp2voll = float(clp2vollintersect/clp2volltotalcl)
    #recall
    cll2volp= float(cll2volpintersect/cll2volptotalcl)
    return ((2*cll2volp*clp2voll)/(clp2voll+cll2volp))
    


ALL_METRICS = {
    "Dice": dice,
    "Precision": precision,
    "Recall": recall,
    "clDice": skiClDice,
    "clRecall": skiClRecall,
    "clPrecision": skiClPrecision
}

# -------------------------------------- evaluator

class Evaluator:
    """Object that holds test and reference segmentations with label information
    and computes a number of metrics on the two. 'labels' must either be an
    iterable of numeric values (or tuples thereof) or a dictionary with string
    names and numeric values.
    """

    default_metrics = [
        "Dice",
        "Precision",
        "Recall",
        "clDice",
        "clRecall",
        "clPrecision"
    ]

    def __init__(self,
                 test=None,
                 reference=None,
                 labels=None,
                 metrics=None,
                 nan_for_nonexisting=True):

        self.test = None
        self.reference = None
        self.testcl = None
        self.referencecl = None
        self.confusion_matrix = ConfusionMatrix()
        self.labels = None
        self.nan_for_nonexisting = nan_for_nonexisting
        self.result = None

        self.metrics = []
        if metrics is None:
            for m in self.default_metrics:
                self.metrics.append(m)
        else:
            for m in metrics:
                self.metrics.append(m)

        self.set_reference(reference)
        self.set_test(test)
        if labels is not None:
            self.set_labels(labels)
        else:
            if test is not None and reference is not None:
                self.construct_labels()

    def set_test(self, test):
        """Set the test segmentation."""

        self.test = test

    def set_reference(self, reference):
        """Set the reference segmentation."""

        self.reference = reference

    def set_testcl(self, testcl):
        """Set the test segmentation."""

        self.testcl = testcl

    def set_referencecl(self, referencecl):
        """Set the reference segmentation."""

        self.referencecl = referencecl

    def set_labels(self, labels):
        """Set the labels.
        :param labels= may be a dictionary (int->str), a set (of ints), a tuple (of ints) or a list (of ints). Labels
        will only have names if you pass a dictionary"""

        if isinstance(labels, dict):
            self.labels = collections.OrderedDict(labels)
        elif isinstance(labels, set):
            self.labels = list(labels)
        elif isinstance(labels, np.ndarray):
            self.labels = [i for i in labels]
        elif isinstance(labels, (list, tuple)):
            self.labels = labels
        else:
            raise TypeError("Can only handle dict, list, tuple, set & numpy array, but input is of type {}".format(type(labels)))

    def construct_labels(self):
        """Construct label set from unique entries in segmentations."""

        if self.test is None and self.reference is None:
            raise ValueError("No test or reference segmentations.")
        elif self.test is None:
            labels = np.unique(self.reference)
        else:
            labels = np.union1d(np.unique(self.test),
                                np.unique(self.reference))
        self.labels = list(map(lambda x: int(x), labels))

    def set_metrics(self, metrics):
        """Set evaluation metrics"""

        if isinstance(metrics, set):
            self.metrics = list(metrics)
        elif isinstance(metrics, (list, tuple, np.ndarray)):
            self.metrics = metrics
        else:
            raise TypeError("Can only handle list, tuple, set & numpy array, but input is of type {}".format(type(metrics)))

    def add_metric(self, metric):

        if metric not in self.metrics:
            self.metrics.append(metric)

    def evaluate(self, test=None, reference=None, **metric_kwargs):
        """Compute metrics for segmentations."""
        if test is not None:
            self.set_test(test)

        if reference is not None:
            self.set_reference(reference)

        if self.test is None or self.reference is None:
            raise ValueError("Need both test and reference segmentations.")

        if self.labels is None:
            self.construct_labels()

        self.metrics.sort()

        # get functions for evaluation
        # somewhat convoluted, but allows users to define additonal metrics
        # on the fly, e.g. inside an IPython console
        _funcs = {m: ALL_METRICS[m] for m in self.metrics}
        frames = inspect.getouterframes(inspect.currentframe())
        for metric in self.metrics:
            for f in frames:
                if metric in f[0].f_locals:
                    _funcs[metric] = f[0].f_locals[metric]
                    break
            else:
                if metric in _funcs:
                    continue
                else:
                    raise NotImplementedError(
                        "Metric {} not implemented.".format(metric))

        # get results
        self.result = OrderedDict()

        eval_metrics = self.metrics

        if isinstance(self.labels, dict):

            for label, name in self.labels.items():
                k = str(name)
                self.result[k] = OrderedDict()
                if not hasattr(label, "__iter__"):
                    self.confusion_matrix.set_test(self.test == label)
                    self.confusion_matrix.set_reference(self.reference == label)
                    self.confusion_matrix.set_skirefcl(skeletonize(self.reference)==label)
                    self.confusion_matrix.set_skitestcl(skeletonize(self.test)==label)
                else:
                    current_test = 0
                    current_reference = 0
                    for l in label:
                        current_test += (self.test == l)
                        current_reference += (self.reference == l)
                    self.confusion_matrix.set_test(current_test)
                    self.confusion_matrix.set_reference(current_reference)
                for metric in eval_metrics:
                    self.result[k][metric] = _funcs[metric](confusion_matrix=self.confusion_matrix,
                                                               nan_for_nonexisting=self.nan_for_nonexisting,
                                                               **metric_kwargs)

        else:

            for i, l in enumerate(self.labels):
                k = str(l)
                self.result[k] = OrderedDict()
                self.confusion_matrix.set_test(self.test == l)
                self.confusion_matrix.set_reference(self.reference == l)
                self.confusion_matrix.set_skirefcl(skeletonize(self.reference)==l)
                self.confusion_matrix.set_skitestcl(skeletonize(self.test)==l)
                for metric in eval_metrics:
                    self.result[k][metric] = _funcs[metric](confusion_matrix=self.confusion_matrix,
                                                            nan_for_nonexisting=self.nan_for_nonexisting,
                                                            **metric_kwargs)

        return self.result

    def to_dict(self):

        if self.result is None:
            self.evaluate()
        return self.result

    def to_array(self):
        """Return result as numpy array (labels x metrics)."""

        if self.result is None:
            self.evaluate

        result_metrics = sorted(self.result[list(self.result.keys())[0]].keys())

        a = np.zeros((len(self.labels), len(result_metrics)), dtype=np.float32)

        if isinstance(self.labels, dict):
            for i, label in enumerate(self.labels.keys()):
                for j, metric in enumerate(result_metrics):
                    a[i][j] = self.result[self.labels[label]][metric]
        else:
            for i, label in enumerate(self.labels):
                for j, metric in enumerate(result_metrics):
                    a[i][j] = self.result[label][metric]

        return a


class NiftiEvaluator(Evaluator):

    def __init__(self, *args, **kwargs):

        self.test_nifti = None
        self.reference_nifti = None
        self.test_nifti_cl = None
        self.reference_nifti_cl = None
        super(NiftiEvaluator, self).__init__(*args, **kwargs)

    def set_test(self, test):
        """Set the test segmentation."""

        if test is not None:
            self.test_nifti = sitk.ReadImage(test)
            super(NiftiEvaluator, self).set_test(sitk.GetArrayFromImage(self.test_nifti))
        else:
            self.test_nifti = None
            super(NiftiEvaluator, self).set_test(test)

    def set_reference(self, reference):
        """Set the reference segmentation."""

        if reference is not None:
            self.reference_nifti = sitk.ReadImage(reference)
            super(NiftiEvaluator, self).set_reference(sitk.GetArrayFromImage(self.reference_nifti))
        else:
            self.reference_nifti = None
            super(NiftiEvaluator, self).set_reference(reference)

    def evaluate(self, test=None, reference=None, voxel_spacing=None, **metric_kwargs):

        if voxel_spacing is None:
            voxel_spacing = np.array(self.test_nifti.GetSpacing())[::-1]
            metric_kwargs["voxel_spacing"] = voxel_spacing

        return super(NiftiEvaluator, self).evaluate(test, reference, **metric_kwargs)


def run_evaluation(args):
    test, ref, evaluator, metric_kwargs = args
    # evaluate
    evaluator.set_test(test)
    evaluator.set_reference(ref)
    if evaluator.labels is None:
        evaluator.construct_labels()
    current_scores = evaluator.evaluate(**metric_kwargs)
    if type(test) == str:
        current_scores["test"] = test
    if type(ref) == str:
        current_scores["reference"] = ref
    return current_scores


def aggregate_scores(test_ref_pairs,
                     evaluator=NiftiEvaluator,
                     labels=None,
                     nanmean=True,
                     num_threads=2,
                     **metric_kwargs):


    if type(evaluator) == type:
        evaluator = evaluator()

    if labels is not None:
        evaluator.set_labels(labels)

    all_scores = OrderedDict()
    all_scores["all"] = []
    all_scores["mean"] = OrderedDict()

    test = [i[0] for i in test_ref_pairs]
    ref = [i[1] for i in test_ref_pairs]
    p = Pool(num_threads)
    all_res = p.map(run_evaluation, zip(test, ref, [evaluator]*len(ref), [metric_kwargs]*len(ref)))
    p.close()
    p.join()

    for i in range(len(all_res)):
        all_scores["all"].append(all_res[i])

        # append score list for mean
        for label, score_dict in all_res[i].items():
            if label in ("test", "reference"):
                continue
            if label not in all_scores["mean"]:
                all_scores["mean"][label] = OrderedDict()
            
            for score, value in score_dict.items():
                if score not in all_scores["mean"][label]:
                    all_scores["mean"][label][score] = []
                 
                all_scores["mean"][label][score].append(value)
                

    for label in all_scores["mean"]:
        for score in all_scores["mean"][label]:
            if nanmean:
                all_scores["mean"][label][score] = float(np.nanmean(all_scores["mean"][label][score]))
            else:
                all_scores["mean"][label][score] = float(np.mean(all_scores["mean"][label][score]))

    return {"Dice": all_scores["mean"]["1"]["Dice"], "Re": all_scores["mean"]["1"]["Recall"], "Pr": all_scores["mean"]["1"]["Precision"],
        "clDice": all_scores["mean"]["1"]["clDice"], "clRe": all_scores["mean"]["1"]["clRecall"], "clPr": all_scores["mean"]["1"]["clPrecision"]}


def evaluate_folder(folder_with_gts: str, folder_with_predictions: str, labels: tuple, **metric_kwargs):
    """
    folder_with_gts: folder where the ground truth segmentations are saved. Must be nifti files .nii.gz
    folder_with_predictions: folder where the predicted segmentations are saved. Must be nifti files .nii.gz
    """
    l = lambda x, y: y
    files_gt = [l(folder_with_gts, i) for i in os.listdir(folder_with_gts) if os.path.isfile(os.path.join(folder_with_gts, i))
            and (i.endswith(".nii.gz"))]
    files_gt.sort()
    files_pred = [l(folder_with_predictions, i) for i in os.listdir(folder_with_predictions) if os.path.isfile(os.path.join(folder_with_predictions, i))
            and (i.endswith(".nii.gz"))]
    files_pred.sort()
    test_ref_pairs = [(os.path.join(folder_with_predictions, i), os.path.join(folder_with_gts, i)) for i in files_pred]
    res = aggregate_scores(test_ref_pairs, num_threads=1, labels=labels, **metric_kwargs)
    print(res)
    return res


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-ref', required=True, type=str, help="Folder containing the reference segmentations in .nii.gz")
    parser.add_argument('-pred', required=True, type=str, help="Folder containing the predicted segmentations in .nii.gz")
    args = parser.parse_args()
    return evaluate_folder(args.ref, args.pred, labels=(1,))


if __name__ == "__main__":
    main()
