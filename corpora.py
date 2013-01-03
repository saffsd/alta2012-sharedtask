"""
Hydrat interface to ALTA2012 Shared Task on EBM Abstracts

Marco Lui, September 2012
"""
import os, csv

from numpy.random.mtrand import RandomState
from hydrat import config
from hydrat.configuration import Configurable, DIR

from hydrat.dataset.text import DirPerClass, ByteUBT, ByteQuadgram, BytePentagram 
from hydrat.dataset.encoded import CodepointUBT, UTF8, ASCII, Latin1
from hydrat.dataset.words import NLTKWord
from hydrat.dataset.split import CrossValidation, LearnCurve, TrainTest

from hydrat.wrapper.treetagger import TreeTagger
from collections import defaultdict
from itertools import groupby
import numpy as np
import hydrat.common.extractors as ext

CLASSES = ['background','intervention','other','outcome','population','study design']

class AutoSplit(CrossValidation, LearnCurve, TrainTest):
  def sp_crossvalidation(self):
    return self.crossvalidation('ebmcat', 10, RandomState(61383441363))

  def sp_learncurve(self):
    return self.learncurve('ebmcat', 0.1, 10, RandomState(61383441363))

  def sp_traintest(self):
    return self.traintest('ebmcat', 9, RandomState(61383441363))


class ALTA2012Common(AutoSplit, NLTKWord, TreeTagger, ByteUBT, ByteQuadgram, BytePentagram, Latin1):
  def _parse_data(self):
    raise NotImplementedError('deriving class to implement')

  def identifiers(self):
    self._parse_data()
    return sorted(self._data['ts'].keys())

  def ts_byte(self):
    self._parse_data()
    return self._data['ts']

  def sq_abstract(self):
    """
    Sequencing information for sentences in a single abstract
    """
    ids = sorted(self.identifiers())
    retval = []
    for k, g in groupby(ids, lambda x: x.split('-')[0]):
      seq = sorted(g, key=lambda x: int(x.split('-')[1]))
      if len(seq) > 1:
        retval.append(seq)
     
    return retval

  def fm_positionabs(self):
    """
    Absolute position in terms of sentence number
    """
    docids = self.identifiers()
    fm = {}
    for docid in docids:
      fm[docid] = {'pos':int(docid.split('-')[1])}
    return fm

  def fm_positionrel(self):
    """
    relative position by index
    """
    docids = self.identifiers()
    sqs = self.sequence('abstract')
    sq_sizes = {}
    for sq in sqs:
      abstract_id = sq[0].split('-')[0]
      sq_sizes[abstract_id] = int(sq[-1].split('-')[1])

    retval = {}
    for docid in docids:
      abstract_id = docid.split('-')[0]
      sent_pos = int(docid.split('-')[1])
      sq_size = sq_sizes.get(abstract_id,1) # number of sents in abs
      relpos = (sent_pos - 1) / float(sq_size)
      retval[docid] = {'relpos':relpos}
    return retval

  def fm_positionrelbyte(self):
    """
    relative position by byte count
    """
    docids = self.identifiers()
    sqs = self.sequence('abstract')
    tss = self.tokenstream('byte')
    sq_sizes = {}
    for sq in sqs:
      abstract_id = sq[0].split('-')[0]
      sizes = [float(len(tss[x])) for x in sq]
      sq_sizes[abstract_id] = np.cumsum(sizes) / np.sum(sizes)

    retval = {}
    for docid in docids:
      abstract_id = docid.split('-')[0]
      # -2 as -1 for 1-indexed, and -1 for index value before.
      sent_pos = int(docid.split('-')[1]) - 2
      if sent_pos == -1:
        # First item, define position as 0
        retval[docid] = {'relposbyte':0.}
      else:
        sq_size = sq_sizes.get(abstract_id)
        val = float(sq_size[sent_pos])
        retval[docid] = {'relposbyte':val}
    return retval

  def fm_sentlenabs(self):
    """
    Absolute sentence length in bytes
    """
    tss = self.tokenstream('byte')
    retval = {}
    for docid in tss:
      retval[docid] = {'sentlenabs':len(tss[docid])}
    return retval


  def fm_abstractlenabs(self):
    """
    Absolute length of abstract in bytes
    """
    docids = self.identifiers()
    sqs = self.sequence('abstract')
    tss = self.tokenstream('byte')

    # Compute the total size of each sequence in bytes
    sq_sizes = {}
    for sq in sqs:
      abstract_id = sq[0].split('-')[0]
      size = sum(len(tss[x]) for x in sq)
      sq_sizes[abstract_id] = size

    retval = {}
    for docid in tss:
      abslen = sq_sizes.get(docid.split('-')[0])
      if abslen is None:
        # single-sentence abstract, the doclen is the abslen
        abslen = len(tss[docid])
      retval[docid] = {'abstractlenabs':abslen}
    return retval

  def fm_sentlenrel(self):
    """
    Relative sentence length in ratio
    """
    docids = self.identifiers()
    sqs = self.sequence('abstract')
    tss = self.tokenstream('byte')

    # Compute the total size of each sequence in bytes
    sq_sizes = {}
    for sq in sqs:
      abstract_id = sq[0].split('-')[0]
      size = sum(len(tss[x]) for x in sq)
      sq_sizes[abstract_id] = size

    retval = {}
    for docid in tss:
      doclen = len(tss[docid])
      abslen = sq_sizes.get(docid.split('-')[0])
      if abslen is None:
        # Single-sentence abstract, no associated sequence, so the sentence 
        # is the whole thing
        retval[docid] = {'sentlenrel':1.0}
      else:
        retval[docid] = {'sentlenrel':float(doclen) / abslen}
    return retval

  def _headings(self):
    """
    Returns a mapping from abstract ID onto a list of previous-heading
    per-sentence
    """

    sqs = self.sequence('abstract')
    tss = self.tokenstream('byte')

    _headings = {}
    for sq in sqs:
      abstract_id = sq[0].split('-')[0]
      heading = 'NONE'
      heading_list = []
      for s in sq:
        sent = tss[s]
        if is_heading(sent):
          heading = sent
        heading_list.append(heading)
      _headings[abstract_id] = heading_list
    return _headings
     

  def fm_headingord(self):
    """
    Section heading, as an ordinal value. A value of 0 indicates no previous heading.
    A heading sentence is considered as labelling itself.
    """
    docids = self.identifiers()

    heading_index = {}
    heading_index['NONE'] = 0

    headings = self._headings()

    retval = {}
    for docid in docids:
      abstract_id = docid.split('-')[0]
      if abstract_id in headings:
        sent_id = int(docid.split('-')[1]) - 1
        heading = headings[abstract_id][sent_id]
        if heading not in heading_index:
          heading_index[heading] = len(heading_index)
        retval[docid] = {'headingord':heading_index[heading]}
      else:
        # Single line abstract, no heading.
        retval[docid] = {'headingord':heading_index['NONE']}
    return retval

  def fm_headingvec(self):
    """
    Section heading, as a boolean vector
    """
    docids = self.identifiers()

    headings = self._headings()

    retval = {}
    for docid in docids:
      abstract_id = docid.split('-')[0]
      if abstract_id in headings:
        sent_id = int(docid.split('-')[1]) - 1
        heading = headings[abstract_id][sent_id]
        retval[docid] = {heading:1}
      else:
        # Single line abstract, no heading.
        retval[docid] = {'NONE':1}
    return retval

  def fm_headingprev(self):
    """
    Frequency vector of headings prior to this point.
    For a document [A,A,A,B,C], this is {A:3,B:1,C:1}
    """
    docids = self.identifiers()
    headings = self._headings()

    # Compute distribution over previous headings for 
    # each docid
    retval = {}
    for abstract_id in headings:
      abs_dist = defaultdict(int)
      for i,heading in enumerate(headings[abstract_id]):
        abs_dist[heading] += 1
        docid = '{0}-{1}'.format(abstract_id, i+1)
        retval[docid] = dict(abs_dist)

    # Patch in values for single-sentence abstracts that
    # are not represented in sq_abstract
    for docid in docids:
      if docid not in retval:
        # Single line abstract, no heading.
        retval[docid] = {'NONE':1}
    return retval



  def fm_bowprev1(self):
    return self.bowoffset([-1])

  def fm_bowprev2(self):
    return self.bowoffset([-2,-1])

  def fm_bowprev3(self):
    return self.bowoffset(range(-3,0))

  def fm_bowprev4(self):
    return self.bowoffset(range(-4,0))

  def fm_bowprev5(self):
    return self.bowoffset(range(-5,0))

  def fm_bowprev6(self):
    return self.bowoffset(range(-6,0))


  def fm_bowpost1(self):
    return self.bowoffset([1])

  def fm_bowpost2(self):
    return self.bowoffset([1,2])

  def fm_bowpost3(self):
    return self.bowoffset([1,2,3])

  def fm_bowpost4(self):
    return self.bowoffset(range(1,5))

  def fm_bowpost5(self):
    return self.bowoffset(range(1,6))

  def fm_bowpost6(self):
    return self.bowoffset(range(1,7))



  def fm_bowwindow1(self):
    return self.bowoffset([-1,0,1])

  def fm_bowwindow2(self):
    return self.bowoffset([-2,-1,0,1,2])

  def fm_bowwindow3(self):
    return self.bowoffset([-3,-2,-1,0,1,2,3])




  def bowoffset(self, offsets):
    """
    Compute a bag-of-words for given offsets
    The offsets are relative. 0 is this item -1 is
    the previous in the sequence and 1 is the next. Where
    the item doesn't exist (e.g. after end of sequence) the
    offset is simply ignored.
    """
    docids = self.identifiers()
    sqs = self.sequence('abstract')
    bow = self.featuremap('nltkword_unigram')

    # Compute distribution given offsets for each docid
    retval = {}
    for sq in sqs:
      abstract_id = sq[0].split('-')[0]
      for i,docid in enumerate(sq):
        tot_bow = defaultdict(int)
        for offset in offsets:
          off_id = '{0}-{1}'.format(abstract_id,i+1+offset)
          if off_id in bow:
            # only process if we actually have this offset
            off_bow = bow[off_id]
            for key in off_bow:
              tot_bow[key] += off_bow[key]
        retval[docid] = dict(tot_bow)

    # Patch in values for single-sentence abstracts that
    # are not represented in sq_abstract
    for docid in docids:
      if docid not in retval:
        if 0 in offsets:
          retval[docid] = bow[docid]
        else:
          retval[docid] = {}
    return retval



  def fm_headingprev1(self):
    return self.headingoffset([-1])

  def fm_headingprev2(self):
    return self.headingoffset(range(-2,0))

  def fm_headingpost1(self):
    return self.headingoffset([1])

  def fm_headingpost2(self):
    return self.headingoffset(range(1,3))

  def fm_headingwindow1(self):
    return self.headingoffset([-1,0,1])

  def fm_headingwindow2(self):
    return self.headingoffset(range(-2,3))



  def headingoffset(self, offsets):
    """
    Compute a headings distribution for given offsets.
    The offsets are relative. 0 is this item -1 is
    the previous in the sequence and 1 is the next. Where
    the item doesn't exist (e.g. after end of sequence) the
    offset is simply ignored.
    """
    docids = self.identifiers()
    headings = self._headings()

    # Compute distribution given offsets for each docid
    retval = {}
    for abstract_id, head_sq in headings.iteritems():
      for i in range(len(head_sq)):
        docid = '{0}-{1}'.format(abstract_id, i+1)
        tot_dist = defaultdict(int)
        for offset in offsets:
          try:
            off_heading = head_sq[i+1+offset]
            tot_dist[off_heading] += 1
          except IndexError:
            # Do nothing if we dont have that index
            pass
        retval[docid] = dict(tot_dist)

    # Patch in values for single-sentence abstracts that
    # are not represented in sq_abstract
    for docid in docids:
      if docid not in retval:
        if 0 in offsets:
          retval[docid] = {'NONE':1}
        else:
          retval[docid] = {}
    return retval

  def fm_bowprev(self):
    return self.prev('nltkword', ext.unigram)

  def fm_bowpost(self):
    return self.post('nltkword', ext.unigram)

  def fm_ttbprev(self):
    return self.prev('treetaggerpos', ext.bigram)

  def fm_ttbpost(self):
    return self.post('treetaggerpos', ext.bigram)

  def fm_ttlbprev(self):
    return self.prev('treetaggerlemmapos', ext.bigram)

  def fm_ttlbpost(self):
    return self.post('treetaggerlemmapos', ext.bigram)

  def prev(self, tsname, extractor):
    docids = self.identifiers()
    sqs = self.sequence('abstract')
    bow = self.features(tsname, extractor)

    # Compute distribution over previous headings for 
    # each docid
    retval = {}
    for sq in sqs:
      abstract_id = sq[0].split('-')[0]
      abs_dist = defaultdict(int)
      for i,docid in enumerate(sq):
        doc_bow = bow[docid]
        for key in doc_bow:
          abs_dist[key] += doc_bow[key]
        retval[docid] = dict(abs_dist)

    # Patch in values for single-sentence abstracts that
    # are not represented in sq_abstract
    for docid in docids:
      if docid not in retval:
        # Single line abstract, so no previous BoW
        retval[docid] = {}
    return retval


  def post(self, tsname, extractor):
    docids = self.identifiers()
    sqs = self.sequence('abstract')
    bow = self.features(tsname, extractor)

    # Compute distribution over previous headings for 
    # each docid
    retval = {}
    for sq in sqs:
      abstract_id = sq[0].split('-')[0]
      abs_dist = defaultdict(int)
      for i,docid in reversed(list(enumerate(sq))):
        doc_bow = bow[docid]
        for key in doc_bow:
          abs_dist[key] += doc_bow[key]
        retval[docid] = dict(abs_dist)

    # Patch in values for single-sentence abstracts that
    # are not represented in sq_abstract
    for docid in docids:
      if docid not in retval:
        # Single line abstract, so no previous BoW
        retval[docid] = {}
    return retval

  def fm_headingpost(self):
    """
    Headings after this point. Like headingprev, we include 
    the current heading.
    """
    docids = self.identifiers()
    headings = self._headings()

    # Compute distribution over subsequent headings for 
    # each docid
    retval = {}
    for abstract_id in headings:
      abs_dist = defaultdict(int)
      for i,heading in reversed(list(enumerate(headings[abstract_id]))):
        abs_dist[heading] += 1
        docid = '{0}-{1}'.format(abstract_id, i+1)
        retval[docid] = dict(abs_dist)

    # Patch in values for single-sentence abstracts that
    # are not represented in sq_abstract
    for docid in docids:
      if docid not in retval:
        # Single line abstract, no heading.
        retval[docid] = {'NONE':1}
    return retval

  def fm_headingprevEXC(self):
    """
    Frequency vector of headings prior to this point.
    For a document [A,A,A,B,C], this is {A:3,B:1,C:1}
    """
    docids = self.identifiers()
    headings = self._headings()

    # Compute distribution over previous headings for 
    # each docid
    retval = {}
    for abstract_id in headings:
      abs_dist = defaultdict(int)
      for i,heading in enumerate(headings[abstract_id]):
        docid = '{0}-{1}'.format(abstract_id, i+1)
        retval[docid] = dict(abs_dist)
        abs_dist[heading] += 1 # this takes the model before adding the current

    # Patch in values for single-sentence abstracts that
    # are not represented in sq_abstract
    for docid in docids:
      if docid not in retval:
        # Single line abstract, no heading.
        #retval[docid] = {'NONE':1}
        retval[docid] = {}
    return retval

  def fm_isstructured(self):
    """
    Feature for describing whether the sentence is part of a structured
    abstract. We use the presence/absence of headings to determine this.
    """
    docids = self.identifiers()
    headings = self._headings()

    retval = {}
    for docid in docids:
      abstract_id = docid.split('-')[0]
      if abstract_id in headings:
        if any(h != 'NONE' for h in headings[abstract_id]):
          retval[docid] = {'structured':1}
        else:
          retval[docid] = {'unstructured':1}
      else:
        retval[docid] = {'unstructured':1}

    return retval


class ALTA2012(ALTA2012Common):
  requires={ 
    ('corpora', 'alta2012-ebm') : DIR('corpus10k'),
    }
  _data = None

  def _parse_data(self):
    if self._data is None:
      cm = defaultdict(list)
      ts = {}
      with open(os.path.join(config.getpath('corpora','alta2012-ebm'),'train.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
          docid = row['Document'] + '-' + row['Sentence']
          if row['Prediction'] == '1':
            cm[docid].append(row['Label'])
          ts[docid] = row['Text']

      self._data = {}
      self._data['cm'] = dict(cm)
      self._data['ts'] = dict(ts)

  def cm_ebmcat(self):
    self._parse_data()
    return self._data['cm']


class ALTA2012Eval(ALTA2012Common):
  requires={ 
    ('corpora', 'alta2012-ebm') : DIR('corpus10k'),
    }
  _data = None

  def _parse_data(self):
    if self._data is None:
      ts = {}
      with open(os.path.join(config.getpath('corpora','alta2012-ebm'),'test.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
          docid = row['Document'] + '-' + row['Sentence']
          ts[docid] = row['Text']

      self._data = {}
      self._data['ts'] = dict(ts)

class ALTA2012Full(ALTA2012Common):
  requires={ 
    ('corpora', 'alta2012-ebm') : DIR('corpus10k'),
    }
  _data = None

  def _parse_data(self):
    if self._data is None:
      # We clobber together the data from the train, test and separate goldstandard files
      train = ALTA2012()
      train._parse_data()
      train_data = train._data
      self.trainids = train_data['ts'].keys()

      test = ALTA2012Eval()
      test._parse_data()
      test_data = test._data
      self.testids = test_data['ts'].keys()

      ts = dict(train_data['ts'])
      ts.update(test_data['ts'])
      self._data = {}
      self._data['ts'] = ts
  
  def cm_ebmcat(self):
    path = os.path.join(config.getpath('corpora','alta2012-ebm'),'GS', 'gs1.txt')
    cm = {}
    with open(path) as f:
      reader = csv.reader(f, delimiter='\t')
      for row in reader:
        key = '{0}-{1}'.format(row[0], row[1])
        value = row[2].split(',')
        cm[key] = value
    return cm

  def sp_crossvalidation(self):
    sq_index = dict( (s[0].split('-')[0],s) for s in self.sequence('abstract'))
    sp = {}
    with open(os.path.join(config.getpath('corpora','alta2012-ebm'),'data.testset')) as f:
      for i,row in enumerate(f):
        key = 'fold{0}'.format(i)
        value = []
        for abs_id in row.split('\t',1)[1].split(':'):
          if abs_id.strip():
            try:
              value.extend(sq_index[abs_id])
            except KeyError:
              # No sequence, single-sentence abstract
              value.append('{0}-1'.format(abs_id))
        sp[key] = value
    return sp

  def sp_traintest(self):
    # train/test based on the split provided for the competition
    sp = {'train':self.trainids, 'test':self.testids}

    return sp

    

def counts2dist(counts):
  tot = float(sum(counts.values()))
  retval = dict( (k,'{0:.1f}%'.format(v/tot*100)) for k,v in counts.iteritems())
  return retval

def is_heading(sent):
  return (sent.upper() == sent) and ('A' <= sent[0] <= 'Z')

if __name__ == "__main__":
  from collections import defaultdict, Counter
  count = defaultdict(Counter)
  st_map = {}
  sent_count = Counter()
  x = ALTA2012Full()
  isstructured = x.featuremap('isstructured')
  cm = x.classmap('ebmcat')
  ts = x.tokenstream('byte')
  labcount = defaultdict(Counter)

  headcount = 0
  for key in isstructured:
    absid = key.split('-')[0]
    st = isstructured[key].keys()[0]
    st_map[absid] = st
    # count by sentence
    sent_count[st] += 1
    cl = cm[key]
    labcount[st][len(cl)] += 1
    for klass in cl:
      # count by label
      count[st][klass] += 1

    if is_heading(ts[key]):
      headcount += 1

  print "NUM ABS", len(st_map)
  print "ABS BREAKDOWN", Counter(st_map.values())
  print "ST COUNT", sent_count
  print "HEADCOUNT", headcount

    
  print "S COUNT", (count['structured'])
  print "S DIST", counts2dist(count['structured'])
  print "U COUNT", (count['unstructured'])
  print "U DIST", counts2dist(count['unstructured'])
  scount = sum(count['structured'].values())
  ucount = sum(count['unstructured'].values())
  print 'structured', scount
  print 'unstructured', ucount
  print 'total', scount+ucount

  print 'MULTILABEL', labcount
  import ipdb;ipdb.set_trace()
