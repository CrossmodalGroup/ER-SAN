from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils
from functools import reduce
import copy
import math
import numpy as np
from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper
from misc.utils import expand_feats
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class TransformerEnsemble(CaptionModel):
    def __init__(self, models, opt):
        super(TransformerEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.vocab_size = models[0].vocab_size
        self.seq_length = models[0].seq_length
        self.ss_prob = 0
        self.n = len(self.models)

    def get_logprobs_state(self, it, memory, mask, state):
        """
        state = [ys.unsqueeze(0)]
        """
        logprobs_ensemble = []
        state_ensemble = []
        for i in range(self.n):
            logprobs_i, state_i = self.models[i].get_logprobs_state(it, memory[i], mask, state)
            logprobs_ensemble.append(logprobs_i.unsqueeze(0))
            state_ensemble.append(state_i)
        logprobs_out = torch.mean(torch.cat(logprobs_ensemble, 0), dim=0)
        state_out = state_ensemble[0]
        return logprobs_out, state_out

    def _forward(self, sg_data, att_feats, boxes, labels, att_masks):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i](sg_data, att_feats, boxes, labels, att_masks)
            out_ensemble.append(out_i.unsqueeze(0))
        outputs = torch.mean(torch.cat(out_ensemble, 0), dim=0)
        return outputs

    def _sample_beam(self, sg_data, att_feats, boxes, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = att_feats.size(0)

        En_memory = []
        for i in range(self.n):
            temp_mem, temp_mask = self.models[i].GET_EncoderData(sg_data, att_feats, boxes, att_masks)
            En_memory.append(temp_mem)
            if i == 0:
                En_att_masks = temp_mask
            elif not En_att_masks.equal(temp_mask):
                En_att_masks = temp_mask
                print('!!!!memory not the same')

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = None
            tmp_memory = [En_memory[i][k:k+1].expand(*((beam_size,)+En_memory[i].size()[1:])).contiguous() for i,m in enumerate(self.models)]
            tmp_att_masks = En_att_masks[k:k+1].expand(*((beam_size,)+En_att_masks.size()[1:])).contiguous() if En_att_masks is not None else None

            for t in range(1):
                if t == 0: # input <bos>
                    it = att_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_memory, tmp_att_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_memory, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        #return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)


    def _sample(self, sg_data, att_feats, boxes, att_masks=None, opt={},_core_args=None):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        return_core_args = opt.get('return_core_args', True)
        expand_features = opt.get('expand_features', False)
        batch_size = att_feats.size(0)
        if beam_size > 1:
            return self._sample_beam(sg_data, att_feats, boxes, att_masks, opt)

        state = None
        En_memory = []
        for i in range(self.n):
            temp_mem, temp_mask = self.models[i].GET_EncoderData(sg_data, att_feats, boxes, att_masks)
            En_memory.append(temp_mem)
            if i == 0:
                En_att_masks = temp_mask
            elif not En_att_masks.equal(temp_mask):
                En_att_masks = temp_mask
                print('!!!!memory not the same')


        seq = att_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = att_feats.new_zeros(batch_size, self.seq_length)

        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = att_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, En_memory, En_att_masks, state)
            if decoding_constraint and t > 0:
                tmp = output.new_zeros(output.size(0), self.vocab_size + 1)
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        returns = [seq, seqLogprobs]
        if return_core_args:
            returns.append(_core_args)
        return returns

    def beam_search(self, init_state, init_logprobs, *args, **kwargs):

        # function computes the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[prev_labels]] - diversity_lambda
            return unaug_logprobsf

        # does one step of classical beam search

        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam

            ys,ix = torch.sort(logprobsf,1,True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols): # for each column (word, essentially)
                for q in range(rows): # for each beam expansion
                    #compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q,c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    local_unaug_logprob = unaug_logprobsf[q,ix[q,c]]
                    candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':local_unaug_logprob})
            candidates = sorted(candidates,  key=lambda x: -x['p'])

            new_state = [_.clone() for _ in state]
            #beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
            #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                #fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                #rearrange recurrent states
                for state_ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
            state = new_state
            return beam_seq,beam_seq_logprobs,beam_logprobs_sum,state,candidates

        # Start diverse_beam_search
        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        max_ppl = opt.get('max_ppl', 0)
        bdash = beam_size // group_size # beam per group

        # INITIALIZATIONS
        beam_seq_table = [torch.LongTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(bdash) for _ in range(group_size)]

        # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
        done_beams_table = [[] for _ in range(group_size)]
        state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state).chunk(group_size, 2)]
        logprobs_table = list(init_logprobs.chunk(group_size, 0))
        # END INIT

        # Chunk elements in the args
        args = list(args)
        args_tmp = []
        args_list_tmp = []
        for _ in args:
            if type(_) != list:
                args_tmp.append(_.chunk(group_size) if _ is not None else [None]*group_size)
            else:
                args_list_tmp.append([i.chunk(group_size) if i is not None else [None]*group_size for i in _])

        args_tmp = [[args_tmp[i][j] for i in range(len(args_tmp))] for j in range(group_size)]
        args_list_tmp = [[[args_list_tmp[j][i][k] for i in range(len(self.models))] for j in range(len(args_list_tmp))] for k in range(group_size)]
        args = []
        for i in range(group_size):
            args.append(args_list_tmp[i] + args_tmp[i])

        for t in range(self.seq_length + group_size - 1):
            for divm in range(group_size):
                if t >= divm and t <= self.seq_length + divm - 1:
                    # add diversity
                    logprobsf = logprobs_table[divm].data.float()
                    # suppress previous word
                    if decoding_constraint and t-divm > 0:
                        logprobsf.scatter_(1, beam_seq_table[divm][t-divm-1].unsqueeze(1).cuda(), float('-inf'))
                    # suppress UNK tokens in the decoding
                    logprobsf[:,logprobsf.size(1)-1] = logprobsf[:, logprobsf.size(1)-1] - 1000
                    # diversity is added here
                    # the function directly modifies the logprobsf values and hence, we need to return
                    # the unaugmented ones for sorting the candidates in the end. # for historical
                    # reasons :-)
                    unaug_logprobsf = add_diversity(beam_seq_table,logprobsf,t,divm,diversity_lambda,bdash)

                    # infer new beams
                    beam_seq_table[divm],\
                    beam_seq_logprobs_table[divm],\
                    beam_logprobs_sum_table[divm],\
                    state_table[divm],\
                    candidates_divm = beam_step(logprobsf,
                                                unaug_logprobsf,
                                                bdash,
                                                t-divm,
                                                beam_seq_table[divm],
                                                beam_seq_logprobs_table[divm],
                                                beam_logprobs_sum_table[divm],
                                                state_table[divm])

                    # if time's up... or if end token is reached then copy beams
                    for vix in range(bdash):
                        if beam_seq_table[divm][t-divm,vix] == 0 or t == self.seq_length + divm - 1:
                            final_beam = {
                                'seq': beam_seq_table[divm][:, vix].clone(),
                                'logps': beam_seq_logprobs_table[divm][:, vix].clone(),
                                'unaug_p': beam_seq_logprobs_table[divm][:, vix].sum().item(),
                                'p': beam_logprobs_sum_table[divm][vix].item()
                            }
                            if max_ppl:
                                final_beam['p'] = final_beam['p'] / (t-divm+1)
                            done_beams_table[divm].append(final_beam)
                            # don't continue beams from finished sequences
                            beam_logprobs_sum_table[divm][vix] = -1000

                    # move the current group one step forward in time

                    it = beam_seq_table[divm][t-divm]
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it.cuda(), *(args[divm] + [state_table[divm]]))

        # all beams are sorted by their log-probabilities
        done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
        done_beams = reduce(lambda a,b:a+b, done_beams_table)
        return done_beams