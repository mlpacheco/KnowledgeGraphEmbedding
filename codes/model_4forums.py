#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import average_precision_score, f1_score, classification_report
from transformers import BertModel, BertConfig

from torch.utils.data import DataLoader
from dataloader import FourForumsDataset

# 4Forums constants
HAS_STANCE = 0
DISAGREE = 1
AGREE = 2

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity_raw, nentity_sym, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity_raw = nentity_raw
        self.nentity_sym = nentity_sym
        self.nrelation = nrelation
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim

        # Define encoder for the raw text entities
        bert_config = BertConfig.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.bert2hidden = nn.Linear(bert_config.hidden_size, self.entity_dim)

        # Keep traditional embedding lookup for symbolic entities
        self.entity_embedding = nn.Parameter(torch.zeros(nentity_sym, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def get_entity_raw(self, text, mask, segment):
        #print(text.size(), mask.size(), segment.size())
        outputs = self.bert_model(text, attention_mask=mask, token_type_ids=segment,
                                  position_ids=None, head_mask=None)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.bert2hidden(pooled_output)
        return pooled_output

    def forward(self, sample, text, mask, segment, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            # I am assuming that the batch size is 1
            relation_type = sample[:, 1][0].item()

            if relation_type == HAS_STANCE:
                head = self.get_entity_raw(
                    text[sample[:, 0]],
                    mask[sample[:, 0]],
                    segment[sample[:, 0]]
                ).unsqueeze(1)

                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=sample[:, 2]
                ).unsqueeze(1)

                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=sample[:, 1]
                ).unsqueeze(1)

            else:
                head = self.get_entity_raw(
                    text[sample[:, 0]],
                    mask[sample[:, 0]],
                    segment[sample[:, 0]]
                ).unsqueeze(1)

                tail = self.get_entity_raw(
                    text[sample[:, 2]],
                    mask[sample[:, 2]],
                    segment[sample[:, 2]]
                ).unsqueeze(1)

                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=sample[:, 1]
                ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample

            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            # I am assuming that the batch size is 1
            relation_type = head_part[:, 1][0].item()

            if relation_type == HAS_STANCE:
                head = self.get_entity_raw(
                    text[head_part[:, 0]],
                    mask[head_part[:, 0]],
                    segment[head_part[:, 0]]
                ).unsqueeze(1)

                # Embed tail
                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=head_part[:, 1]
                ).unsqueeze(1)

            else:
                head = self.get_entity_raw(
                    text[head_part[:, 0]],
                    mask[head_part[:, 0]],
                    segment[head_part[:, 0]]
                ).unsqueeze(1)

                tail = self.get_entity_raw(
                    text[tail_part.view(-1)],
                    mask[tail_part.view(-1)],
                    segment[tail_part.view(-1)]
                ).view(batch_size, negative_sample_size, -1)

                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=head_part[:, 1]
                ).unsqueeze(1)
        elif mode == 'reln-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = self.get_entity_raw(
                text[head_part[:, 0]],
                mask[head_part[:, 0]],
                segment[head_part[:, 0]]
            ).unsqueeze(1)

            tail = self.get_entity_raw(
                text[head_part[:, 2]],
                mask[head_part[:, 2]],
                segment[head_part[:, 2]]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, text, mask, segment):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            #subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), text, mask, segment, mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample, text, mask, segment)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2

        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 +
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()
        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples_node, test_triples_link, args, text, mask, segment, other_id):
        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()

        #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        #Prepare dataloader for evaluation
        test_dataloader_node = DataLoader(
            FourForumsDataset(test_triples_node, other_id, is_test=True),
            batch_size = args.batch_size,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=FourForumsDataset.collate_fn
        )
        test_dataloader_link = DataLoader(
            FourForumsDataset(test_triples_link, other_id, is_test=True),
            batch_size = args.batch_size,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=FourForumsDataset.collate_fn
        )
        test_dataset_list = [test_dataloader_node, test_dataloader_link]

        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        agree_pred = []; disagree_pred = []; stance_pred = []
        agree_gold = []; disagree_gold = []; stance_gold = []

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()

                    batch_size = positive_sample.size(0)
                    score = model((positive_sample, negative_sample), text, mask, segment, mode)

                    #Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim = 1, descending=True)

                    if mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    elif mode == 'reln-batch':
                        positive_arg = positive_sample[:, 1]
                    else:
                        raise ValueError('mode %s not supported' % mode)
                    relation_arg = positive_sample[:, 1]

                    for i in range(batch_size):
                        #Notice that argsort is not ranking
                        gold = positive_arg[i].item()
                        prediction = negative_sample[i, :][argsort[i, :][0]].item()
                        relation = relation_arg[i].item()

                        if relation == HAS_STANCE:
                            stance_gold.append(gold)
                            stance_pred.append(prediction)
                        elif relation == AGREE:
                            agree_gold.append(gold)
                            agree_pred.append(prediction)
                        elif relation == DISAGREE:
                            disagree_gold.append(gold)
                            disagree_pred.append(prediction)
                        else:
                            raise ValueError('relation %s not supported' % relation)

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1
        labels = list(set([0,1,2]) - set([other_id]))
        #print(agree_gold)
        #print(agree_pred)
        metrics = {
            "stance_f1": f1_score(stance_gold, stance_pred, average="macro", labels=labels),
            "agree_f1": f1_score(agree_gold, agree_pred, average="binary", pos_label=AGREE),
            "disagree_f1": f1_score(disagree_gold, disagree_pred, average="binary", pos_label=DISAGREE),
        }
        print(classification_report(stance_gold, stance_pred))
        metrics["agree_disagree_f1"] = (metrics["agree_f1"] + metrics["disagree_f1"]) / 2

        return metrics
