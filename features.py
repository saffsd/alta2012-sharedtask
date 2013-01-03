"""
Generate the feature sets
"""
from corpora import ALTA2012Full

import hydrat.common.extractors as ext
from hydrat.proxy import DataProxy
from hydrat.store import Store
from hydrat import config

feats_bow = ['bowprev', 'bowpost', 
      'bowprev1', 'bowpost1', 'bowwindow1',
      'bowprev2', 'bowpost2', 'bowwindow2',
      'bowprev3', 'bowpost3', 'bowwindow3',
      'bowprev4', 'bowpost4', 
      'bowprev5', 'bowpost5',
      'bowprev6', 'bowpost6',
      ]

feats_content = [ 'nltkword_unigram', 'treetaggerpos_bigram', 
                  'treetaggerlemmapos_bigram', 'treetaggerpos_trigram'
                ]

feats_prev = [ 'bowprev', 
               'bowprev1', 'bowprev2', 'bowprev3', 
               'bowprev4', 'bowprev5', 'bowprev6',
               'ttbprev', 'ttlbprev',
               'headingprev', 'headingprev1', 'headingprev2',
      ]

feats_post = [ 'bowpost', 
               'bowpost1', 'bowpost2', 'bowpost3', 
               'bowpost4', 'bowpost5', 'bowpost6',
               'ttbpost', 'ttlbpost',
               'headingpost', 'headingpost1', 'headingpost2',
      ]

feats_window = [ 'bowwindow1', 'bowwindow2', 'bowwindow3',
                 'headingwindow1', 'headingwindow2',
                 ]
               

feats_pos = ['ttbprev', 'ttbpost', 'ttlbprev', 'ttlbpost', 
             'treetaggerpos_bigram', 'treetaggerlemmapos_bigram',
             'treetaggerpos_trigram' ]

feats_struct = [
        'positionabs','positionrel','positionrelbyte',
        'sentlenabs','sentlenrel','isstructured',
        'abstractlenabs',
      ]

feats_heading = ['headingord', 'headingvec', 'headingprev', 'headingprevEXC', 'headingpost',
      'headingprev1','headingpost1','headingwindow1',
      'headingprev2','headingpost2','headingwindow2',
      ]

feats_position = ['positionabs','positionrel','positionrelbyte']


core = ( 'nltkword_unigram', 'treetaggerpos_bigram', 'treetaggerlemmapos_bigram', )
core += ('headingprev', 'headingvec', 'positionrel')
struct_best = core + ('bowpost1', 'bowprev3', 'headingpost', 'isstructured', 'sentlenrel',)
unstruct_best = core + ('bowprev','treetaggerpos_trigram','ttbprev')

feats_all  = tuple(sorted(set(sum(map(tuple, [struct_best, unstruct_best, feats_bow, feats_pos, feats_struct, feats_heading, feats_position]),tuple()))))

datasets = [
  ALTA2012Full(),
  ]

if __name__ == "__main__":
  store = Store.from_caller()

  for ds in datasets:
    proxy = DataProxy(ds, store=store)
    proxy.inducer.process(proxy.dataset, 
      fms=feats_all,
      cms=['ebmcat',], 
      sqs=['abstract',],
    )

    proxy.tokenstream_name = 'treetaggerlemmapos'
    proxy.tokenize(ext.bigram)

    proxy.tokenstream_name = 'treetaggerpos'
    proxy.tokenize(ext.bigram)
    proxy.tokenize(ext.trigram)
