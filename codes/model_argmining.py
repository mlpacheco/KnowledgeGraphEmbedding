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

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

from torch.utils.data import DataLoader
from dataloader import ArgMiningDataset

# ArgMining Constants
MAJOR_CLAIM = 6089
CLAIM = 6090
PREMISE = 6091
SUPPORT = 6092
ATTACK = 6093
NONE = 6094
IS_ARG_TYPE = 0
RELN = 1
STANCE = 2

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity_raw, nentity_sym, nrelation, hidden_dim, gamma,
                 vocab_size, word_embed_size,
                 double_entity_embedding=False, double_relation_embedding=False, indnets=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity_raw = nentity_raw
        self.nentity_sym = nentity_sym
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
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
        self.word_embeds = nn.Embedding(vocab_size, word_embed_size)
        self.word_bilstm = nn.LSTM(input_size=word_embed_size,
                                   hidden_size=128,
                                   bidirectional=True,
                                   batch_first=True)
        self.word_layer = nn.Linear(128*2 + 34, self.entity_dim)

        self.indnets = indnets
        if not self.indnets:
            # Define the layers for the relation encoder
            #self.reln_layer = nn.Linear(self.entity_dim + self.entity_dim + 1014, self.relation_dim)
            self.reln_layer = nn.Linear(1014, self.relation_dim)
        else:
            self.c1_word_embeds = nn.Embedding(vocab_size, word_embed_size)
            self.c1_word_bilstm = nn.LSTM(input_size=word_embed_size,
                                          hidden_size=128,
                                          bidirectional=True,
                                          batch_first=True)
            self.c1_word_layer = nn.Linear(128*2 + 34, self.entity_dim)

            self.c2_word_embeds = nn.Embedding(vocab_size, word_embed_size)
            self.c2_word_bilstm = nn.LSTM(input_size=word_embed_size,
                                          hidden_size=128,
                                          bidirectional=True,
                                          batch_first=True)
            self.c2_word_layer = nn.Linear(128*2 + 34, self.entity_dim)
            self.reln_layer = nn.Linear(self.entity_dim*2 + 1014, 128)

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

    def load_pretrained_word_embed(self, pretrained_embeddings):
        self.word_embeds.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def init_hidden_bilstm(self):
        var1 = torch.autograd.Variable(torch.zeros(2, self.minibatch_size, 128))
        var2 = torch.autograd.Variable(torch.zeros(2, self.minibatch_size, 128))
        var1 = var1.cuda()
        var2 = var2.cuda()
        return (var1, var2)

    def _get_entity_raw(self, x, seq_lengths, x_feats, word_embeds, word_bilstm, word_layer):
        self.minibatch_size = len(x)

        # Sort according to lengths
        seq_len_sorted, sorted_idx = seq_lengths.sort(descending=True)
        # sort inputs
        x = x[sorted_idx]
        x = word_embeds(x)

        # pack padded sequences
        packed_input_seq = pack_padded_sequence(x, list(seq_len_sorted.data), batch_first=True)
        # run lstm over sequence
        self.hidden_bilstm = self.init_hidden_bilstm()
        packed_output, self.hidden_bilstm = \
            word_bilstm(packed_input_seq, self.hidden_bilstm)
        # unpack the output
        unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # Reverse sorting
        unpacked_output = torch.zeros_like(unpacked_output).scatter_(0, sorted_idx.unsqueeze(1).unsqueeze(1).expand(-1, unpacked_output.shape[1], unpacked_output.shape[2]), unpacked_output)
        # Do global avg pooling over timesteps
        lstm_output = torch.mean(unpacked_output, dim=1)

        # Add extra features
        lstm_output = torch.cat([lstm_output, x_feats], 1)
        lstm_output = F.relu(word_layer(lstm_output))

        return lstm_output

    def get_entity_raw(self, x, seq_lengths, x_feats):
        return self._get_entity_raw(x, seq_lengths, x_feats, self.word_embeds, self.word_bilstm, self.word_layer)

    def get_relation_raw(self, reln_feats):
    #def get_relation_raw(self, head, tail, reln_feats):
        #if (head.size()[1] < tail.size()[1]):
        #    head_repeat = head.repeat(1, tail.size()[1], 1)
        #    concat = torch.cat([head_repeat, tail, reln_feats], 2)
            #print(head_repeat.size(), tail.size(), reln_feats.size())
        #else:
            #print(head.size(), tail.size(), reln_feats.size())
        #    concat = torch.cat([head, tail, reln_feats], 2)
        #concat = self.reln_layer(concat)
        concat = self.reln_layer(reln_feats)
        concat = F.relu(concat)
        return concat

    def get_relation_raw_indnets(self, c1, c1_lengths, c1_feats, c2, c2_lengths, c2_feats, reln_feats):
        c1_output = self._get_entity_raw(c1, c1_lengths, c1_feats, self.c1_word_embeds, self.c1_word_bilstm, self.c1_word_layer)
        c2_output = self._get_entity_raw(c2, c2_lengths, c2_feats, self.c2_word_embeds, self.c2_word_bilstm, self.c2_word_layer)
        c1_output = c1_output.unsqueeze(1)
        c2_output = c2_output.view(-1, c2_output.size()[0], c2_output.size()[1])

        if (c1_output.size()[1] < c2_output.size()[1]):
            c1_output = c1_output.repeat(1, c2_output.size()[1], 1)

        concat = torch.cat([c1_output, c2_output, reln_feats], 2)
        concat = self.reln_layer(concat)
        concat = F.relu(concat)
        return concat

    def forward(self, sample, id2text, seq_lengths, id2features, reln_features, mode='single'):
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

            if relation_type == IS_ARG_TYPE or relation_type == STANCE:
                head = self.get_entity_raw(
                    id2text[sample[:, 0]],
                    seq_lengths[sample[:, 0]],
                    id2features[sample[:, 0]]
                ).unsqueeze(1)
                # Use offset for mapping symbolic entities to parameters
                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=sample[:, 2] - self.nentity_raw
                ).unsqueeze(1)
                relation = torch.index_select(
                    self.relation_embedding,
                    dim=0,
                    index=sample[:, 1]
                ).unsqueeze(1)
            else:
                head = self.get_entity_raw(
                    id2text[sample[:, 0]],
                    seq_lengths[sample[:, 0]],
                    id2features[sample[:, 0]]
                ).unsqueeze(1)
                tail = self.get_entity_raw(
                    id2text[sample[:, 2]],
                    seq_lengths[sample[:, 2]],
                    id2features[sample[:, 2]]
                ).unsqueeze(1)

                reln_features = reln_features.unsqueeze(1)
                if not self.indnets:
                    #relation = self.get_relation_raw(head, tail, reln_features)
                    relation = self.get_relation_raw(reln_features)

                else:
                    relation = self.get_relation_raw_indnets(
                        c1=id2text[sample[:, 0]],
                        c1_lengths=seq_lengths[sample[:, 0]],
                        c1_feats=id2features[sample[:, 0]],
                        c2=id2text[sample[:, 2]],
                        c2_lengths=seq_lengths[sample[:, 2]],
                        c2_feats=id2features[sample[:, 2]],
                        reln_feats=reln_features
                    )

        elif mode == 'tail-batch':
            head_part, tail_part = sample

            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            # I am assuming that the batch size is 1
            relation_type = head_part[:, 1][0].item()

            if relation_type == IS_ARG_TYPE or relation_type == STANCE:
                head = self.get_entity_raw(
                    id2text[head_part[:, 0]],
                    seq_lengths[head_part[:, 0]],
                    id2features[head_part[:, 0]]
                ).unsqueeze(1)

                # Use offset for mapping symbolic entities to parameters
                tail_part = tail_part - self.nentity_raw
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

            elif relation_type == RELN:
                head = self.get_entity_raw(
                    id2text[head_part[:, 0]],
                    seq_lengths[head_part[:, 0]],
                    id2features[head_part[:, 0]]
                ).unsqueeze(1)

                tail = self.get_entity_raw(
                    id2text[tail_part.view(-1)],
                    seq_lengths[tail_part.view(-1)],
                    id2features[tail_part.view(-1)]
                ).view(batch_size, negative_sample_size, -1)

                if not self.indnets:
                    #relation = self.get_relation_raw(head, tail, reln_features)
                    relation = self.get_relation_raw(reln_features)
                else:
                    relation = self.get_relation_raw_indnets(
                        c1=id2text[head_part[:, 0]],
                        c1_lengths=seq_lengths[head_part[:, 0]],
                        c1_feats=id2features[head_part[:, 0]],
                        c2=id2text[tail_part.view(-1)],
                        c2_lengths=seq_lengths[tail_part.view(-1)],
                        c2_feats=id2features[tail_part.view(-1)],
                        reln_feats=reln_features
                    )

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
    def train_step(model, optimizer, train_iterator, args, id2text, seq_lengths, id2features):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        #positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        positive_sample, negative_sample, mode, positive_feats, negative_feats = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            positive_feats = positive_feats.cuda()
            negative_feats = negative_feats.cuda()
            #subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), id2text, seq_lengths, id2features, negative_feats, mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample, id2text, seq_lengths, id2features, positive_feats)

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

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, graph, entity2subgraph, args, id2text, seq_lengths, id2features, id2entity, data, indicators, prod_rules):
        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()

        #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        #Prepare dataloader for evaluation
        test_dataloader = DataLoader(
            ArgMiningDataset(test_triples, graph, entity2subgraph, id2entity=id2entity, data=data, indicators=indicators, prod_rules=prod_rules, is_test=True),
            batch_size = args.batch_size,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=ArgMiningDataset.collate_fn
        )
        logs = []

        step = 0
        total_steps = len(test_dataloader)

        argument_pred = []; argument_gold = []
        stance_pred = []; stance_gold = []
        reln_pred = []; reln_gold = []

        with torch.no_grad():
            for positive_sample, negative_sample, mode, _, negative_feats in test_dataloader:
                if args.cuda:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    negative_feats = negative_feats.cuda()

                batch_size = positive_sample.size(0)
                score = model((positive_sample, negative_sample), id2text, seq_lengths, id2features, negative_feats, mode)

                #Explicitly sort all the entities to ensure that there is no test exposure bias
                argsort = torch.argsort(score, dim = 1, descending=True)

                #if mode == 'head-batch':
                #    positive_arg = positive_sample[:, 0]
                if mode == 'tail-batch':
                    positive_arg = positive_sample[:, 2]
                else:
                    raise ValueError('mode %s not supported' % mode)
                relation_arg = positive_sample[:, 1]

                for i in range(batch_size):
                    #Notice that argsort is not ranking
                    gold = positive_arg[i].item()


                    prediction = negative_sample[i, :][argsort[i, :][0]].item()
                    relation = relation_arg[i].item()
                    if relation == IS_ARG_TYPE:
                        argument_gold.append(gold)
                        argument_pred.append(prediction)
                    elif relation == STANCE:
                        stance_gold.append(gold)
                        stance_pred.append(prediction)
                    else:
                        #Ranking to evaluate cutoff
                        pos_arg = (negative_sample[i, :] == gold).nonzero()[0][0]
                        #print(negative_sample[i, :])
                        #print(pos_arg)
                        ranking = (argsort[i, :] == pos_arg).nonzero()
                        #print(argsort[i, :])
                        #print(ranking)
                        #exit()
                        assert ranking.size(0) == 1
                        ranking = 1 + ranking.item()

                        #print(negative_sample[i, :])
                        #print(gold, prediction)
                        for index, candidate in enumerate(negative_sample[i, :]):
                            gold_label = int(candidate.item() == gold)
                            #pred_label = int(candidate.item() == prediction)
                            #reln_pred.append(pred_label)
                            curr_ranking = (argsort[i, :] == index).nonzero().item() + 1
                            if curr_ranking <= 9:
                                reln_pred.append(1)
                            else:
                                reln_pred.append(0)
                            reln_gold.append(gold_label)


                        #print(reln_gold)
                        #print(reln_pred)
                        #exit()
                        #ranking + 1 is the true ranking used in evaluation metrics
                        logs.append({
                            'MRR': 1.0/ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@2': 1.0 if ranking <= 2 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@4': 1.0 if ranking <= 4 else 0.0,
                            'HITS@5': 1.0 if ranking <= 5 else 0.0
                            #'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = {
            "argument_f1": f1_score(argument_gold, argument_pred, average="macro"),
            "stance_f1": f1_score(stance_gold, stance_pred, average="macro", labels=[SUPPORT, ATTACK]),
            "reln_f1": f1_score(reln_gold, reln_pred, pos_label=1, average="binary"),
            "max_rank": max([log['MR'] for log in logs])
        }
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
