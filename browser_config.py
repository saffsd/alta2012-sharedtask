classes = ['background','intervention','outcome','population','study design','other']

from hydrat.summary import classification_summary, ClassPRF
summary_fn = classification_summary()
for kl in classes:
  summary_fn.extend(ClassPRF(kl))

from hydrat.result.interpreter import NonZero, SingleHighestValue
#interpreter = NonZero()
interpreter = SingleHighestValue()

from hydrat.summary import Summary
from collections import defaultdict
class BaseFeatures(Summary):
  def key_basefeatures(self): 
    retval = defaultdict(int)
    # TODO: Handle if feature_desc is absent
    for feat in self.result.metadata['feature_desc']:
      # TODO: Handle more complex feature desc
      retval[feat] = 1
    return retval

  def key_basefeaturecount(self):
    return len(self.result.metadata['feature_desc'])

from hydrat.datamodel import Result
from hydrat.result import CombinedMicroAverage 
import numpy as np
class AbstractTypePRF(Summary):
  """
  Breakdown of results by abstract type
  """
  def key_abstype(self):
    if hasattr(self.result, 'store'):
      # have a store, can look stuff up
      feat_name = 'isstructured'
      ds = self.result.metadata['dataset']
      cs = self.result.metadata['class_space']
      fm = self.result.store.get_FeatureMap(ds, feat_name)
      feat_space = self.result.store.get_Space(feat_name)

      # Extract goldstandard and overall classification matrix
      # TODO: Check that CL has full coverage w/o overlap
      gs = self.result.store.get_ClassMap(ds, cs).raw
      cl = np.zeros(gs.shape, dtype=self.result.results[0].classifications.dtype)
      for r in self.result.results:
        cl[r.instance_indices] = r.classifications

      retval = {}
      for i, cat in enumerate(feat_space):
        cat_mask = fm[:,i].nonzero()[0]
        cat_res = Result(gs[cat_mask], cl[cat_mask], cat_mask)
        cm = cat_res.confusion_matrix(self.interpreter)
        prf = CombinedMicroAverage(cm[None,:])
        retval[cat] = {'precision':prf.precision, 'recall':prf.recall, 'fscore':prf.fscore}
      return retval
    else:
      return {}

from hydrat.result import PRF
def prf_dict(confmat):
  return dict(zip(('precision','recall','fscore'),PRF()(confmat)))

class Breakdown(Summary):
  """
  Breakdown of results using a space
  """
  def __init__(self, space_name):
    """
    @param space_name name of space to break down by
    """
    # TODO: make the ConfusionMatrixMetric a parameter which influences the key naming
    # TODO: configurable macro/microaverage 
    Summary.__init__(self)
    self.space_name = space_name
    self.cmm = prf_dict
  
  def key_breakdown(self):
    if hasattr(self.result, 'store'):
      store = self.result.store
      # have a store, can look stuff up
      ds = self.result.metadata['dataset']
      cs = self.result.metadata['class_space']
      try:
        class_space = store.get_Space(cs)
        space = store.get_Space(self.space_name)
        space_md = store.get_SpaceMetadata(self.space_name)
      except NoData:
        # TODO: handle NoData here with an error message
        raise

      # get the actual space and check it is suitable
      if space_md['type'] == 'feature':
        fm = store.get_FeatureMap(ds, self.space_name).raw.toarray()
        if fm.min() < 0 or fm.max() > 1:
          raise TypeError('feature space has non 0/1 elements')
        mat = fm.astype(bool)
      elif space_md['type'] == 'class':
        mat = store.get_ClassMap(ds, self.space_name).raw
      else:
        raise TypeError('space {0} has unsuitable type: {1}'.format(self.space_name, space_md['type']))

      # Extract goldstandard and overall classification matrix
      # TODO: Check that CL has full coverage w/o overlap
      gs = self.result.store.get_ClassMap(ds, cs).raw
      cl = np.zeros(gs.shape, dtype=self.result.results[0].classifications.dtype)
      for r in self.result.results:
        cl[r.instance_indices] = r.classifications

      retval = {}
      for i, cat in enumerate(space):
        cat_mask = mat[:,i].nonzero()[0]
        cat_res = Result(gs[cat_mask], cl[cat_mask], cat_mask)
        cm = cat_res.confusion_matrix(self.interpreter)
        prf = self.cmm(cm.sum(0))
        breakdown = dict(zip(class_space, map(self.cmm,cm) ))
        retval[cat] = (prf, breakdown)
      return retval
    else:
      return {}


class AblatedFeats(Summary):
  def __init__(self, all_feats):
    Summary.__init__(self)
    self.all_feats = set(all_feats)

  def init(self, result, interpreter):
    self.md = result.metadata

  def key_ablated(self):
    return tuple(sorted(self.all_feats - set(self.md['feature_desc'])))

from features import feats_all
summary_fn.extend(BaseFeatures())
summary_fn.extend(AblatedFeats(feats_all))
summary_fn.extend(Breakdown("isstructured"))

relevant  = [
  #( {'label':"Dataset", 'searchable':True}       , "dataset"       ),
  #( {'label':"Class Space",'searchable':True}     , "class_space"     ),
  #( {'label':"BoW",'searchable':False}   , "basefeatures:nltkword_unigram"     ),
  #( {'label':"Position",'searchable':False}   , "basefeatures:positionrel"     ),
  #( {'label':"Heading",'searchable':False}   , "basefeatures:headingprev"     ),
  #( {'label':"POS 2-gram",'searchable':False}   , "basefeatures:treetaggerpos_bigram"     ),
  #( {'label':"Lemma+POS 2-gram",'searchable':False}   , "basefeatures:treetaggerlemmapos_bigram"     ),
  ( {'label':"Feature Count",'searchable':True}   , "basefeaturecount"),
  ( {'label':"Feature Desc",'searchable':True}   , "metadata:feature_desc"     ),
  #( {'label':"Ablated",'searchable':True}   , "ablated"     ),
  ( {'label':"Learner",'searchable':True}    , "metadata:learner"    ),
  #( {'label':"Params",'searchable':True}    , "learner_params"    ),
  #( "Macro-F"       , "macro_fscore"        ),
  #( "Macro-P"     , "macro_precision"     ),
  #( "Macro-R"        , "macro_recall"        ),
  ( "Micro-F"       , "micro_fscore"        ),
  ( "Micro-P"     , "micro_precision"     ),
  ( "Micro-R"        , "micro_recall"        ),
  ( "S "       , "breakdown:structured:0:fscore"        ),
  ( "U "       , "breakdown:unstructured:0:fscore"        ),
  #( "S-B "       , "breakdown:structured:1:background:fscore"        ),
  #( "S-I "       , "breakdown:structured:1:intervention:fscore"        ),
  #( "S-O "       , "breakdown:structured:1:outcome:fscore"        ),
  #( "S-P "       , "breakdown:structured:1:population:fscore"        ),
  #( "S-S "       , "breakdown:structured:1:study design:fscore"        ),
  #( "S-Ot "      , "breakdown:structured:1:other:fscore"        ),
  #( "U-B "       , "breakdown:unstructured:1:background:fscore"        ),
  #( "U-I "       , "breakdown:unstructured:1:intervention:fscore"        ),
  #( "U-O "       , "breakdown:unstructured:1:outcome:fscore"        ),
  #( "U-P "       , "breakdown:unstructured:1:population:fscore"        ),
  #( "U-S "       , "breakdown:unstructured:1:study design:fscore"        ),
  #( "U-Ot "      , "breakdown:unstructured:1:other:fscore"        ),
  #( {'sorter':'digit', 'label':"Learn Time"}    , "avg_learn"     ),
  #( {'sorter':'digit', 'label':"Classify Time"} , "avg_classify"  ),
]
