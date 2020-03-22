import json
import re

DATA = "/scratch1/pachecog/DRaiL/examples/argument_mining/data/data_updated.json"
IN_PAR = "/scratch1/pachecog/DRaiL/examples/argument_mining/data/in_par.txt"
INDICATORS = "/scratch1/pachecog/DRaiL/examples/argument_mining/data/indicators.json"

def context_feats(essay_id, component_id, n_par, data):
    preceding_text = data[essay_id][str(n_par)][component_id]['preceding_text_par'].lower()
    following_text = data[essay_id][str(n_par)][component_id]['following_text_par'].lower()

    has_forward_follows = int(re.search(indicators["forward"], following_text) is not None)
    has_backward_follows = int(re.search(indicators["backward"], following_text) is not None)
    has_thesis_follows = int(re.search(indicators["thesis"], following_text) is not None)
    has_rebuttal_follows = int(re.search(indicators["rebuttal"], following_text) is not None)

    has_forward_precedes = int(re.search(indicators["forward"], preceding_text) is not None)
    has_backward_precedes = int(re.search(indicators["backward"], preceding_text) is not None)
    has_thesis_precedes = int(re.search(indicators["thesis"], preceding_text) is not None)
    has_rebuttal_precedes = int(re.search(indicators["rebuttal"], preceding_text) is not None)

    np_intro = set(data[essay_id]['np_intro'])
    vp_intro = set(data[essay_id]['vp_intro'])
    np_concl = set(data[essay_id]['np_concl'])
    vp_concl = set(data[essay_id]['vp_concl'])
    np_curr = set(data[essay_id][str(n_par)][component_id]['text_np'])
    vp_curr = set(data[essay_id][str(n_par)][component_id]['text_vp'])

    np_overlap_intro = int(len(np_intro & np_curr) > 0)
    np_overlap_intro_count = len(np_intro & np_curr)
    np_overlap_concl = int(len(np_concl & np_curr) > 0)
    np_overlap_concl_count = len(np_concl & np_curr)

    vp_overlap_intro = int(len(vp_intro & vp_curr) > 0)
    vp_overlap_intro_count = len(vp_intro & vp_curr)
    vp_overlap_concl = int(len(vp_concl & vp_curr) > 0)
    vp_overlap_concl_count = len(vp_concl & vp_curr)

    return [has_forward_follows, has_backward_follows, has_thesis_follows, has_rebuttal_follows,
            has_forward_precedes, has_backward_precedes, has_thesis_precedes, has_rebuttal_precedes,
            np_overlap_intro, np_overlap_concl, np_overlap_intro_count, np_overlap_concl_count,
            vp_overlap_intro, vp_overlap_concl, vp_overlap_intro_count, vp_overlap_concl_count]

def indicator_feats(essay_id, component_id, n_par, data):
    text = data[essay_id][str(n_par)][component_id]['text'].lower()
    preceeding_text = data[essay_id][str(n_par)][component_id]['preceding_text'].lower()

    has_forward = int(re.search(indicators["forward"], text) is not None or re.search(indicators["forward"], preceeding_text) is not None)
    has_backward = int(re.search(indicators["backward"], text) is not None or re.search(indicators["backward"], preceeding_text) is not None)
    has_thesis = int(re.search(indicators["thesis"], text) is not None or re.search(indicators["thesis"], preceeding_text) is not None)
    has_rebuttal = int(re.search(indicators["rebuttal"], text) is not None or re.search(indicators["rebuttal"], preceeding_text) is not None)

    first_person = int(re.search('(^|\s)(i|me|my|mine|myself)($|\s)', text) is not None or
                       re.search('(^|\s)(i|me|my|mine|myself)($|\s)', preceeding_text) is not None)

    return [has_forward, has_backward, has_thesis, has_rebuttal, first_person]

def structural_feats(essay_id, component_id, n_par, data):
    curr_offset = data[essay_id][str(n_par)][component_id]['offset_l']
    other_offsets = [data[essay_id][str(n_par)][cid]['offset_l'] \
                     for cid in data[essay_id][str(n_par)]]

    min_offset = min(other_offsets)
    max_offset = max(other_offsets)

    is_component_frst_inpar = 0
    is_component_last_inpar = 0
    if curr_offset < min_offset:
        is_component_frst_inpar = 1
    if curr_offset > max_offset:
        is_component_last_inpar = 1

    relative_pos_inpar = curr_offset / (1.0 * max_offset)
    n_preceding_components = sum(x < curr_offset for x in other_offsets)
    n_following_components = sum(x > curr_offset for x in other_offsets)
    last_par = data[essay_id]['num_pars'] - 1
    is_component_in_intro = int(n_par == 1)
    is_component_in_concl = int(n_par == last_par)

    n_tokens = len(data[essay_id][str(n_par)][component_id]['tokens'])
    n_tokens_cover_par = data[essay_id][str(n_par)][component_id]['n_tokens_cover_par']
    n_tokens_cover_sent = data[essay_id][str(n_par)][component_id]['n_tokens_cover_sent']
    component_sentence_ratio = n_tokens / (1.0 * n_tokens_cover_sent)
    n_tokens_left = data[essay_id][str(n_par)][component_id]['n_tokens_left']
    n_tokens_right = data[essay_id][str(n_par)][component_id]['n_tokens_right']

    return [is_component_frst_inpar, is_component_last_inpar, relative_pos_inpar,
            n_preceding_components, n_following_components, is_component_in_intro,
            is_component_in_concl, n_tokens, n_tokens_cover_par, n_tokens_cover_sent,
            component_sentence_ratio, n_tokens_left, n_tokens_right]

with open(DATA) as fp:
    data = json.load(fp)

with open(INDICATORS) as fp:
    indicators = json.load(fp)
    for typ in indicators:
        indicators[typ] = "|".join(indicators[typ]).lower()

tokens_dict = {}
text_dict = {}
features_dict = {}
in_par = {}

with open(IN_PAR) as fp:
    for line in fp:
        essay, component, par = line.strip().split()
        tokens = data[essay][par][component]["tokens"]
        text = data[essay][par][component]["text"]
        entity = "{0}/{1}".format(essay, component)
        in_par[entity] = par
        tokens_dict[entity] = tokens
        text_dict[entity] = text
        features_dict[entity] = structural_feats(essay, component, par, data) + \
                                indicator_feats(essay, component, par, data) + \
                                context_feats(essay, component, par, data)

text_file = open("text.dict", "w")
tokens_file = open("tokens.dict", "w")
features_file = open("features.dict", "w")
entities_file_updated = open("entities.dict", "w")

with open("entities.orig.dict") as fp:
    for line in fp:
        id_, entity = line.strip().split()
        if '/' in entity:
            essay, component = entity.strip().split('/')
            entities_file_updated.write("{0}\t{1}/{2}/{3}\n".format(id_, essay, component, in_par[entity]))
        else:
            entities_file_updated.write("{0}\t{1}\n".format(id_, entity))
        if entity in tokens_dict:
            tokens_file.write("{0}\t{1}\n".format(entity, tokens_dict[entity]))
            text_file.write("{0}\t{1}\n".format(entity, text_dict[entity].encode("utf-8")))
            features_file.write("{0}\t{1}\n".format(entity, features_dict[entity]))

text_file.close()
tokens_file.close()
features_file.close()
entities_file_updated.close()
