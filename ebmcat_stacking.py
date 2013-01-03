"""
TODO Description
"""
import hydrat.classifier.liblinear as liblinear
import hydrat.classifier.meta.repeat as repeat 
from hydrat.common import as_set
from hydrat.store import Store
from hydrat.experiment import Experiment
from hydrat.proxy import StackingProxy

from corpora import ALTA2012Full
from features import struct_best, unstruct_best, feats_bow, feats_pos, feats_struct, feats_heading, feats_position, core, feats_all
from features import feats_prev, feats_post, feats_content, feats_window

feats_local = feats_content + ['headingord', 'headingvec'] + feats_struct
features = [
  feats_all,
  ('nltkword_unigram',),
  feats_content,
  feats_content + ['headingord', 'headingvec'],
  feats_local,
  ['headingord', 'headingvec'] + feats_struct,
  feats_struct,
  feats_local + feats_prev,
  feats_local + feats_post,
  feats_local + feats_window,
  feats_local + feats_prev + feats_post,
  feats_local + feats_prev + feats_post + feats_window,
]

features = [ tuple(sorted(x)) for x in features ]
  
if __name__ == "__main__":
  fallback = Store('store/features.h5' )
  store = Store.from_caller(fallback=fallback)

  learner = repeat.RepeatLearner(liblinear.liblinearL(svm_type=0, output_probability=True))
  ds = ALTA2012Full()

  proxy = StackingProxy(ds, learner, store=store)
  proxy.class_space = 'ebmcat'
  for feats in features:
    print "DOING:", len(feats), feats
    proxy.feature_spaces = feats
    e = Experiment(proxy, learner)
    proxy.store.new_TaskSetResult(e)

