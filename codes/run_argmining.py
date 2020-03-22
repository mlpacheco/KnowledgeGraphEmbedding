#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random
import ast

import numpy as np
import torch

from torch.utils.data import DataLoader

from model_argmining import KGEModel

from dataloader import ArgMiningDataset

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    parser.add_argument('--indnets', action='store_true', default=False)

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    #args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'),
        relation_embedding
    )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)
    graph = {}; entity2subgraph = {}


    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict(); id2entity = dict()
        nentity_sym = 0
        for line in fin:
            eid, entity = line.strip().split('\t')
            if '/' in entity:
                #print(eid, entity)
                subgraph_id, component_id, par = entity.split('/')
                if subgraph_id not in graph:
                    graph[subgraph_id] = []
                graph[subgraph_id].append(int(eid))
                entity2subgraph[int(eid)] = subgraph_id
                # store and drop the paragraph info
                id2entity[int(eid)] = {"par": int(par), "component": component_id, "essay": subgraph_id}
                entity = "{0}/{1}".format(subgraph_id, component_id)
            else:
                print(eid, entity)
                nentity_sym += 1
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'tokens.dict')) as fin:
        max_seq_len = 0
        id2text = dict(); seq_lengths = [0] * (len(entity2id) - nentity_sym)
        for line in fin:
            entity, text = line.strip().split('\t')
            id2text[entity2id[entity]] = ast.literal_eval(text)
            curr_len = len(id2text[entity2id[entity]])
            max_seq_len = max(curr_len, max_seq_len)
            seq_lengths[entity2id[entity]] = curr_len

        id2text_tensor = torch.zeros((len(id2text), max_seq_len)).long()
        for idx, curr_len in enumerate(seq_lengths):
            id2text_tensor[idx, :curr_len] = torch.LongTensor(id2text[idx])
        seq_lengths = torch.LongTensor(seq_lengths)

        if args.cuda:
            id2text_tensor = id2text_tensor.cuda()
            seq_lengths = seq_lengths.cuda()

    with open(os.path.join(args.data_path, 'features.dict')) as fin:
        id2features = torch.zeros(len(entity2id), 34).float()
        for line in fin:
            entity, features = line.strip().split('\t')
            id2features[entity2id[entity]] = torch.FloatTensor(ast.literal_eval(features))

        if args.cuda:
            id2features = id2features.cuda()

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
        print(relation2id)

    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))

    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples

    # Load pretrained word embeddings
    logging.info('Loading pretrained word embeddings...')
    pretrained_embeddings = np.load(os.path.join(args.data_path, 'word_embeddings.npy'))
    logging.info(pretrained_embeddings.shape)

    kge_model = KGEModel(
        model_name=args.model,
        nentity_raw=nentity - nentity_sym,
        nentity_sym = nentity_sym,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        vocab_size=pretrained_embeddings.shape[0],
        word_embed_size=pretrained_embeddings.shape[1],
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        indnets=args.indnets
    )
    kge_model.load_pretrained_word_embed(pretrained_embeddings)

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()

    if args.do_train:
        # TO-DO: receive a parameter depending on task to use a different Dataset
        # (Each task has its particular negative sampling procedure)

        data = json.load(open(os.path.join(args.data_path, "data.json")))
        indicators = json.load(open(os.path.join(args.data_path, "indicators.json")))
        prod_rules = json.load(open(os.path.join(args.data_path, "prod_rules.json")))
        train_dataloader = DataLoader(
            ArgMiningDataset(train_triples, graph, entity2subgraph, id2entity=id2entity, data=data, indicators=indicators, prod_rules=prod_rules),
            batch_size = args.batch_size,
            shuffle = True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=ArgMiningDataset.collate_fn
        )
        train_iterator = iter(train_dataloader)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    # Set valid dataloader as it would be evaluated during training

    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []
        curr_patience = 0; best_valid = -1
        #Training Loop
        #for step in range(init_step, args.max_steps):
        step = init_step
        while 1:
            try:
                log = kge_model.train_step(kge_model, optimizer, train_iterator, args, id2text_tensor, seq_lengths, id2features)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                log = kge_model.train_step(kge_model, optimizer, train_iterator, args, id2text_tensor, seq_lengths, id2features)

            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            '''
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
            '''

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, valid_triples, graph, entity2subgraph, args, id2text_tensor, seq_lengths, id2features, id2entity=id2entity, data=data, indicators=indicators, prod_rules=prod_rules)
                log_metrics('Valid', step, metrics)

                average = 0.0; n_metrics = 0
                for metric in metrics:
                    average += metrics[metric]
                    n_metrics += 1
                average = average / n_metrics
                if average > best_valid:
                    save_variable_list = {
                        'step': step,
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
                    save_model(kge_model, optimizer, save_variable_list, args)
                    best_valid = average
                    curr_patience = 0
                else:
                    curr_patience += 1

            if curr_patience >= args.patience:
                break

            step += 1

    data = json.load(open(os.path.join(args.data_path, "data.json")))
    indicators = json.load(open(os.path.join(args.data_path, "indicators.json")))
    prod_rules = json.load(open(os.path.join(args.data_path, "prod_rules.json")))

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, graph, entity2subgraph, args, id2text_tensor, seq_lengths, id2features, id2entity=id2entity, data=data, indicators=indicators, prod_rules=prod_rules)
        log_metrics('Valid', step, metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, graph, entity2subgraph, args, id2text_tensor, seq_lengths, id2features, id2entity=id2entity, data=data, indicators=indicators, prod_rules=prod_rules)
        log_metrics('Test', step, metrics)

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, graph, entity2subgraph, args, id2text_tensor, seq_lengths, id2features, id2entity=id2entity, data=data, indicators=indicators, prod_rules=prod_rules)
        log_metrics('Test', step, metrics)

if __name__ == '__main__':
    main(parse_args())
