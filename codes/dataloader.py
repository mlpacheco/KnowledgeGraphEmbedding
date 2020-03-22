#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import re
import random
from torch.utils.data import Dataset

class ArgMiningDataset(Dataset):
    def __init__(self, triples, graph, entity2subgraph, id2entity=None, data=None, indicators=None, prod_rules=None, is_test=False):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        #self.count = self.count_frequency(triples)
        #self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.graph = graph
        self.entity2subgraph = entity2subgraph
        self.is_test = is_test
        self.id2entity = id2entity
        self.data = data
        self.indicators = indicators
        for typ in self.indicators:
            self.indicators[typ] = "|".join(self.indicators[typ]).lower()

        prod_rules = prod_rules = [(k, prod_rules[k]) for k in prod_rules]
        prod_rules.sort(reverse=True)
        prod_rules = prod_rules[:500]
        prod_rules = [k for (v,k) in prod_rules]
        self.prod_rules_idx = dict({k:v for (k,v) in zip(prod_rules, range(0, 500))})

        # ArgMining Constants
        self.MAJOR_CLAIM = 6089
        self.CLAIM = 6090
        self.PREMISE = 6091
        self.SUPPORT = 6092
        self.ATTACK = 6093
        self.NONE = 6094
        self.IS_ARG_TYPE = 0
        self.RELN = 1
        self.STANCE = 2

    def extract_reln_indicators(self, essay_id_source, component_id_source, n_par_source,
                                essay_id_target, component_id_target, n_par_target):
        offset_l_source = self.data[essay_id_source][str(n_par_source)][component_id_source]['offset_l']
        offset_r_source = self.data[essay_id_source][str(n_par_source)][component_id_source]['offset_r']

        offset_l_target = self.data[essay_id_target][str(n_par_target)][component_id_target]['offset_l']
        offset_r_target = self.data[essay_id_target][str(n_par_target)][component_id_target]['offset_r']

        if offset_l_source < offset_l_target:
            text_in_between = self.data[essay_id_source]['full_text'][offset_r_source:offset_l_target]
        else:
            text_in_between = self.data[essay_id_source]['full_text'][offset_r_target:offset_l_source]

        #print(essay_id_source, component_id_source, n_par_source)
        #print(essay_id_target, component_id_target, n_par_target)

        has_forward = int(re.search(self.indicators["forward"], text_in_between) is not None)
        has_backward = int(re.search(self.indicators["backward"], text_in_between) is not None)
        has_thesis = int(re.search(self.indicators["thesis"], text_in_between) is not None)
        has_rebuttal = int(re.search(self.indicators["rebuttal"], text_in_between) is not None)
        first_person = int(re.search('(^|\s)(i|me|my|mine|myself)($|\s)', text_in_between) is not None)
        return [has_forward, has_backward, has_thesis, has_rebuttal, first_person]

    def extract_reln_structural(self, essay_id_source, component_id_source, n_par_source,
                                essay_id_target, component_id_target, n_par_target):
        source_offset = self.data[essay_id_source][str(n_par_source)][component_id_source]['offset_l']
        target_offset = self.data[essay_id_target][str(n_par_target)][component_id_target]['offset_l']
        source_sent = self.data[essay_id_source][str(n_par_source)][component_id_source]['curr_sent']
        target_sent = self.data[essay_id_target][str(n_par_target)][component_id_target]['curr_sent']
        n_components_in_between = 0

        n_par_begin = min([n_par_source, n_par_target]); begin_offset = min([source_offset, target_offset])
        n_par_end   = max([n_par_source, n_par_target]); end_offset   = max([source_offset, target_offset])
        for in_betw_par in range(n_par_begin, n_par_end + 1):
            if str(in_betw_par) not in self.data[essay_id_source]:
                continue
            for c_id in self.data[essay_id_source][str(in_betw_par)]:
                offset_l = self.data[essay_id_source][str(in_betw_par)][c_id]['offset_l']
                if offset_l > begin_offset and offset_l < end_offset:
                    n_components_in_between += 1

        same_par = int(n_par_source == n_par_target)
        same_sen = int(source_sent == target_sent)

        target_before_source = int(n_par_target < n_par_source or
                                   target_sent < source_sent or
                                   target_offset < source_offset)

        target_after_source = int(n_par_target > n_par_source or
                                  target_sent > source_sent or
                                  target_offset > source_offset)

        both_in_intro = int(n_par_source == 1 and n_par_target == 1)
        last_par = self.data[essay_id_source]['num_pars'] - 1
        both_in_concl = int(n_par_source == last_par and n_par_target == last_par)
        return [n_components_in_between, same_par, same_sen, target_before_source,
                target_after_source, both_in_intro, both_in_concl]

    def extract_reln_shared_nouns(self, essay_id_source, component_id_source, n_par_source,
                                  essay_id_target, component_id_target, n_par_target):
        np_source = set(self.data[essay_id_source][str(n_par_source)][component_id_source]['text_np'])
        np_target = set(self.data[essay_id_target][str(n_par_target)][component_id_target]['text_np'])
        np_overlap = int(len(np_source & np_target) > 0)
        np_overlap_count = len(np_source & np_target)
        return [np_overlap, np_overlap_count]

    def extract_reln_prod_rules(self, essay_id, component_id, n_par):
        ret = [0] * 500
        for prod_rule in self.data[essay_id][str(n_par)][component_id]['prod_rules']:
            if prod_rule in self.prod_rules_idx:
                ret[self.prod_rules_idx[prod_rule]] = 1
        return ret

    def extract_reln_feats(self, essay_id_source, component_id_source, n_par_source,
                           essay_id_target, component_id_target, n_par_target):
        ret = self.extract_reln_structural(essay_id_source, component_id_source, n_par_source,
                                           essay_id_target, component_id_target, n_par_target) +\
              self.extract_reln_indicators(essay_id_source, component_id_source, n_par_source,
                                           essay_id_target, component_id_target, n_par_target) + \
              self.extract_reln_shared_nouns(essay_id_source, component_id_source, n_par_source,
                                             essay_id_target, component_id_target, n_par_target) + \
              self.extract_reln_prod_rules(essay_id_source, component_id_source, n_par_source) + \
              self.extract_reln_prod_rules(essay_id_target, component_id_target, n_par_target)
        return ret

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        positive_features = []

        head, relation, tail = positive_sample

        negative_sample_list = []
        negative_feature_list = []
        negative_sample_size = 0

        argument_classes = set([self.MAJOR_CLAIM, self.CLAIM, self.PREMISE])
        stance_classes = set([self.SUPPORT, self.ATTACK, self.NONE])

        # The negative sampling procedure depends on the relation type
        if relation == self.IS_ARG_TYPE:
            if not self.is_test:
                rel_classes = list(argument_classes - set([tail]))
            else:
                rel_classes = list(argument_classes)
            for k in rel_classes:
                negative_sample = k
                negative_sample_list.append(negative_sample)
        elif relation == self.STANCE:
            if not self.is_test:
                rel_classes = list(stance_classes - set([tail]))
            else:
                rel_classes = list(stance_classes)
            for k in rel_classes:
                negative_sample = k
                negative_sample_list.append(negative_sample)
        else:
            #print(head, tail)
            #print(self.id2entity[head])
            #print(self.id2entity[tail])
            #print("=======")
            positive_features = \
                self.extract_reln_feats(
                    essay_id_source=self.id2entity[head]["essay"],
                    component_id_source=self.id2entity[head]["component"],
                    n_par_source=self.id2entity[head]["par"],
                    essay_id_target=self.id2entity[tail]["essay"],
                    component_id_target=self.id2entity[tail]["component"],
                    n_par_target=self.id2entity[tail]["par"]
                )

            # Shuffle examples first
            for entity in self.graph[self.entity2subgraph[head]]:
                if entity == head and not self.is_test:
                    continue
                #print(head, entity)
                #exit()
                reln_features = \
                    self.extract_reln_feats(
                        essay_id_source=self.id2entity[head]["essay"],
                        component_id_source=self.id2entity[head]["component"],
                        n_par_source=self.id2entity[head]["par"],
                        essay_id_target=self.id2entity[entity]["essay"],
                        component_id_target=self.id2entity[entity]["component"],
                        n_par_target=self.id2entity[entity]["par"]
                    )
                negative_sample = entity
                negative_sample_list.append(negative_sample)
                negative_feature_list.append(reln_features)

        negative_sample = np.array(negative_sample_list)
        negative_sample = torch.from_numpy(negative_sample)
        negative_features = torch.FloatTensor(negative_feature_list)

        # Reduce the proportion of negative examples when training
        '''
        if relation == self.RELN and not self.is_test:
            excerpt = list(range(0, len(negative_sample_list)))[:2]
            random.shuffle(excerpt)
            negative_sample = negative_sample[excerpt]
            negative_features = negative_features[excerpt]
        '''

        positive_sample = torch.LongTensor(positive_sample)
        positive_features = torch.FloatTensor(positive_features)

        return positive_sample, negative_sample, 'tail-batch',\
               positive_features, negative_features

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        positive_features = torch.stack([_[3] for _ in data], dim=0)
        negative_features = torch.stack([_[4] for _ in data], dim=0)
        mode = data[0][2]
        return positive_sample, negative_sample, mode,\
               positive_features, negative_features

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail

class FourForumsDataset(Dataset):
    def __init__(self, triples, other_id, is_test=False):
        self.triples = triples
        self.len = len(triples)
        self.triple_set = set(triples)
        self.is_test = is_test
        self.other_id = other_id

        # 4Forums constants
        self.HAS_STANCE = 0
        self.DISAGREE = 1
        self.AGREE = 2

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        negative_sample_list = []
        negative_sample_size = 0

        stance_classes = set([0, 1, 2])

        if relation == self.HAS_STANCE:
            if not self.is_test:
                rel_classes = list(stance_classes - set([tail, self.other_id]))
            else:
                rel_classes = list(stance_classes - set([self.other_id]))
            for k in rel_classes:
                negative_sample = k
                negative_sample_list.append(negative_sample)
            mode = 'tail-batch'
        elif relation == self.AGREE:
            if not self.is_test:
                negative_sample = self.DISAGREE
                negative_sample_list.append(negative_sample)
            else:
                for k in [self.AGREE, self.DISAGREE]:
                    negative_sample = k
                    negative_sample_list.append(negative_sample)
            mode = 'reln-batch'
        else:
            if not self.is_test:
                negative_sample = self.AGREE
                negative_sample_list.append(negative_sample)
            else:
                for k in [self.AGREE, self.DISAGREE]:
                    negative_sample = k
                    negative_sample_list.append(negative_sample)
            mode = 'reln-batch'

        negative_sample = np.array(negative_sample_list)
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        return positive_sample, negative_sample, mode

class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_node, dataloader_link):
        self.iterator_node = self.one_shot_iterator(dataloader_node)
        self.iterator_link = self.one_shot_iterator(dataloader_link)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_node)
        else:
            data = next(self.iterator_link)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data

class DebatesDataset(Dataset):
    pass

