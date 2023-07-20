
from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Categorical

from hgnn_models.TFJSPModel_hgnn_sub import GATedge, MLPsim, MLPs
from rule_based_rl_models.TFJSPModel_DQN_Rule_sub import MLPActor, MLPCritic
from env.common_func import get_normalized, norm_disc_rewards, select_vehicle_v2

class TFJSPModel_hgnn(nn.Module):
    def __init__(self,
                 env_paras,
                 model_paras,
                 ):
        super().__init__()
        
        # === initial parameter setup ===
        self.in_size_ma = env_paras["ma_feat_dim"]  # Dimension of the raw feature vectors of machine nodes
        self.in_size_ope = env_paras["ope_feat_dim"]  # Dimension of the raw feature vectors of operation nodes

        self.device = model_paras["device"]
        self.out_size_ma = model_paras["out_size_ma"]  # Dimension of the embedding of machine nodes
        self.out_size_ope = model_paras["out_size_ope"]  # Dimension of the embedding of operation nodes
        self.hidden_size_ope = model_paras["hidden_size_ope"]  # Hidden dimensions of the MLPs
        self.actor_dim = model_paras["actor_in_dim"]  # Input dimension of actor
        self.critic_dim = model_paras["critic_in_dim"]  # Input dimension of critic
        self.n_latent_actor = model_paras["n_latent_actor"]  # Hidden dimensions of the actor
        self.n_latent_critic = model_paras["n_latent_critic"]  # Hidden dimensions of the critic
        self.n_hidden_actor = model_paras["n_hidden_actor"]  # Number of layers in actor
        self.n_hidden_critic = model_paras["n_hidden_critic"]  # Number of layers in critic
        self.action_dim = model_paras["action_dim"]  # Output dimension of actor
        
        self.num_heads = model_paras["num_heads"]
        self.dropout = model_paras["dropout"]
        
        # Machine node embedding
        self.get_machines = nn.ModuleList()
        self.get_machines.append(GATedge((self.in_size_ope, self.in_size_ma), self.out_size_ma, self.num_heads[0],
                                    self.dropout, self.dropout, activation=F.elu))
        for i in range(1,len(self.num_heads)):
            self.get_machines.append(GATedge((self.out_size_ope, self.out_size_ma), self.out_size_ma, self.num_heads[i],
                                    self.dropout, self.dropout, activation=F.elu))

        # Operation node embedding
        self.get_operations = nn.ModuleList()
        self.get_operations.append(MLPs([self.out_size_ma, self.in_size_ope, self.in_size_ope, self.in_size_ope],
                                        self.hidden_size_ope, self.out_size_ope, self.num_heads[0], self.dropout))
        for i in range(len(self.num_heads)-1):
            self.get_operations.append(MLPs([self.out_size_ma, self.out_size_ope, self.out_size_ope, self.out_size_ope],
                                            self.hidden_size_ope, self.out_size_ope, self.num_heads[i], self.dropout))

        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim, self.n_latent_actor, self.action_dim).to(self.device)
        self.critic = MLPCritic(self.n_hidden_critic, self.critic_dim, self.n_latent_critic, 1).to(self.device)
        
        self.MseLoss = nn.MSELoss()
        
    
    def init(self, state, dataset=None, loader=None):
        pass
    
    def act(self, state, memory=None):
        action, act_idx, act_prob = self.forward(state)
        if memory is not None:
            memory.add_action_info(action.transpose(1,0).detach().cpu().numpy(), 
                                    act_idx.detach().cpu().numpy(), 
                                    act_prob.detach().cpu().numpy())
        
        return action, act_prob
    
    def forward(self, state, flag_sample=True, flag_train=True):
        # Get probability of actions and the id of the current operation (be waiting to be processed) of each job
        action_probs, ope_step_batch, _ = self.get_action_prob(state, flag_sample, flag_train=flag_train) # [B, n_mas * n_jobs] | 

        # DRL-S, sampling actions following \pi
        if flag_sample:
            dist = Categorical(action_probs)
            action_indexes = dist.sample()
        # DRL-G, greedily picking actions with the maximum probability
        else:
            action_indexes = action_probs.argmax(dim=1)
        select_act_prob = dist.log_prob(action_indexes).exp().unsqueeze(-1)   # [B, 1]

        # Calculate the machine, job and operation index based on the action index
        mas = (action_indexes / state.mask_job_finish_batch.size(1)).long()     # [B,]
        jobs = (action_indexes % state.mask_job_finish_batch.size(1)).long()    # [B,]
        opes = ope_step_batch[state.batch_idxes, jobs]
        
        # === select vehicle ===
        veh_dict = select_vehicle_v2(state, mas.unsqueeze(1), jobs.unsqueeze(1))
        vehs = veh_dict['veh_id'].long()   # [B,]

        return torch.stack((opes, mas, jobs, vehs), dim=1).t(), action_indexes, select_act_prob

    def get_action_prob(self, state, memories=None, flag_sample=False, flag_train=False):
        '''
        Get the probability of selecting each action in decision-making
        '''
        # Uncompleted instances
        batch_idxes = state.batch_idxes
        
        # === raw features ===
        raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]   # [B, n_opes, ope_feat_dim]
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes] # [B, n_mas, ma_feat_dim]
        raw_vehs = state.feat_vehs_batch.transpose(1, 2)[batch_idxes]   # [B, n_vehs, veh_feat_dim]
        proc_time = state.proc_times_batch[batch_idxes] # [B, n_opes, n_mas]
        trans_time = state.trans_times_batch[batch_idxes]   # [B, n_mas, n_mas]
        # : extract raw_opes with current job 
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)  # [B, n_jobs]
        raw_jobs = raw_opes.gather(1, ope_step_batch[:, :, None].expand(-1, -1, raw_opes.size(2)))  # [B, n_jobs]
        
        # === normalize ===
        nums_opes = state.nums_opes_batch[batch_idxes]
        features = get_normalized(raw_opes, raw_mas, raw_vehs, proc_time, trans_time, \
            flag_sample=True, flag_train=True)
        norm_opes = (deepcopy(features[0]))
        norm_mas = (deepcopy(features[1]))
        norm_vehs = (deepcopy(features[2]))
        norm_proc_time = (deepcopy(features[3]))
        norm_trans_time = (deepcopy(features[4]))
        
        feat_tuple = (features[0], features[1], features[3])
        # L iterations of the HGNN
        for i in range(len(self.num_heads)):
            # First Stage, machine node embedding
            # shape: [len(batch_idxes), num_mas, out_size_ma]
            h_mas = self.get_machines[i](state.ope_ma_adj_batch, state.batch_idxes, feat_tuple)   # [D_b, n_mchs, 8]
            feat_tuple = (feat_tuple[0], h_mas, feat_tuple[2])
            # Second Stage, operation node embedding
            # shape: [len(batch_idxes), max(num_opes), out_size_ope]
            h_opes = self.get_operations[i](state.ope_ma_adj_batch, state.ope_pre_adj_batch, state.ope_sub_adj_batch,
                                            state.batch_idxes, feat_tuple)    # [D_b, max(n_opes), 8]
            feat_tuple = (h_opes, feat_tuple[1], feat_tuple[2])

        # Stacking and pooling
        h_mas_pooled = h_mas.mean(dim=-2)  # shape: [len(batch_idxes), out_size_ma]
        # There may be different operations for each instance, which cannot be pooled directly by the matrix
        if not flag_sample and not flag_train:
            h_opes_pooled = []
            for i in range(len(batch_idxes)):
                h_opes_pooled.append(torch.mean(h_opes[i, :nums_opes[i], :], dim=-2))
            h_opes_pooled = torch.stack(h_opes_pooled)  # shape: [len(batch_idxes), d]
        else:
            h_opes_pooled = h_opes.mean(dim=-2)  # shape: [len(batch_idxes), out_size_ope]

        # Detect eligible O-M pairs (eligible actions) and generate tensors for actor calculation
        jobs_gather = ope_step_batch[..., :, None].expand(-1, -1, h_opes.size(-1))[batch_idxes] # [D_b, n_jobs, out_size_ope (8)]
        h_jobs = h_opes.gather(1, jobs_gather)  # [D_b, n_jobs, out_size_ope]
        
        
        # === Matrix indicating whether processing is possible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1,
                          ope_step_batch[..., :, None].expand(-1, -1, state.ope_ma_adj_batch.size(-1))[batch_idxes])
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, state.proc_times_batch.size(-1), -1)   # [D_b, n_jobs, n_mchs, out_size_ope]
        # print(f"state.proc_times_batch:{state.proc_times_batch}")
        # print(f"h_jobs_padding:{h_jobs_padding.shape}")
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)   # [D_b, n_jobs, n_mchs, out_size_mch (8)]
        # h_mas_pooled: [D_b, out_size_mch] | h_mas_pooled[:, None, None, :]: [D_b, 1, 1, out_size_mch]
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding) # [D_b, n_jobs, n_mchs, out_size_mch (8)]
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)
        # === Matrix indicating whether machine is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(h_jobs_padding[..., 0])
        # Matrix indicating whether job is eligible
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                         state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(h_jobs_padding[..., 0])
        eligible = job_eligible & ma_eligible & (eligible_proc == 1)
        if (~(eligible)).all():
            print("No eligible O-M pair!")
            return
        # Input of actor MLP
        # shape: [len(batch_idxes), num_mas, num_jobs, out_size_ma*2+out_size_ope*2]
        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)  # deprecated
        mask = eligible.transpose(1, 2).flatten(1)  # [D_b, m*n]
        # Get priority index and probability of actions with masking the ineligible actions
        scores = self.actor(h_actions).flatten(1)   # [D_b, m*n]
        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)


        return action_probs, ope_step_batch, h_pooled
    
    def evaluate(self, ope_ma_adj, ope_pre_adj, ope_sub_adj, raw_opes, raw_mas, proc_time,
                 ope_step_batch, eligible, action_idx, flag_sample=False):
        '''
        Input:
            ope_step_batch: [B, n_jobs]
        '''
        batch_idxes = torch.arange(0, ope_ma_adj.size(-3)).long()
        features = (raw_opes, raw_mas, proc_time)

        # L iterations of the HGNN
        for i in range(len(self.num_heads)):
            h_mas = self.get_machines[i](ope_ma_adj, batch_idxes, features)
            features = (features[0], h_mas, features[2])
            h_opes = self.get_operations[i](ope_ma_adj, ope_pre_adj, ope_sub_adj, batch_idxes, features)
            features = (h_opes, features[1], features[2])

        # Stacking and pooling
        h_mas_pooled = h_mas.mean(dim=-2)
        h_opes_pooled = h_opes.mean(dim=-2)

        # Detect eligible O-M pairs (eligible actions) and generate tensors for critic calculation
        h_jobs = h_opes.gather(1, ope_step_batch[:, :, None].expand(-1, -1, h_opes.size(2)))    # [B, n_jobs, out_size_ope]
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, proc_time.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)

        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)
        scores = self.actor(h_actions).flatten(1)   # [B, n_jobs * n_mas]
        mask = eligible.transpose(1, 2).flatten(1)  # [B, n_mas * n_jobs]

        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)
        state_values = self.critic(h_pooled)
        dist = Categorical(action_probs.squeeze())
        action_logprobs = dist.log_prob(action_idx)    # [B,]
        dist_entropys = dist.entropy()
        
        return action_logprobs, state_values.squeeze().double(), dist_entropys

    def update(self, memory, optimizer, minibatch_size,
               gamma, K_epochs, eps_clip, A_coeff, vf_coeff, entropy_coeff):
        
        # === sample transition ===
        old_ope_ma_adj, old_ope_pre_adj, old_ope_sub_adj, \
            old_raw_opes, old_raw_mas, old_raw_vehs, \
            old_proc_time, old_trans_time, \
            old_ope_step_batch, old_eligible, \
            old_rewards, old_is_terminals, \
            old_logprobs, old_action_indexes = memory.all_sample()
        
        old_ope_ma_adj = old_ope_ma_adj.transpose(1,0).flatten(0, 1)    # [B * T, n_opes, n_mas]
        old_ope_pre_adj = old_ope_pre_adj.transpose(1,0).flatten(0, 1)    # [B * T, n_opes, n_opes]
        old_ope_sub_adj = old_ope_sub_adj.transpose(1,0).flatten(0, 1)    # [B * T, n_opes, n_opes]
        
        old_raw_opes = old_raw_opes.transpose(1,0).flatten(0, 1).transpose(1,2)        # [B * T, n_opes, feat_ope_dim]
        old_raw_mas = old_raw_mas.transpose(1,0).flatten(0, 1).transpose(1,2)          # [B * T, n_mas, feat_ma_dim]
        old_raw_vehs = old_raw_vehs.transpose(1,0).flatten(0, 1).transpose(1,2)          # [B * T, n_vehs, feat_veh_dim]
        
        old_proc_time = old_proc_time.transpose(1,0).flatten(0, 1)          # [B * T, n_opes, n_mas]
        old_trans_time = old_trans_time.transpose(1,0).flatten(0, 1)          # [B * T, n_mas, n_mas]
        old_ope_step_batch = old_ope_step_batch.transpose(1,0).flatten(0, 1)    # [B * T, n_jobs]
        old_eligible = old_eligible.transpose(1,0).flatten(0, 1)          # [B * T, n_jobs, n_mas]

        old_rewards = old_rewards.transpose(1,0)                            # [B, T]
        old_is_terminals = old_is_terminals.transpose(1,0)                  # [B, T]
        
        old_logprobs = old_logprobs.transpose(1,0).flatten(0, 1).squeeze()            # [B * T, ]
        old_action_indexes = old_action_indexes.transpose(1,0).flatten(0, 1).squeeze()    # [B * T, ]

        # === normalize discounted rewards ===
        rewards_batch, _ = norm_disc_rewards(old_rewards, old_is_terminals, gamma, self.device)   # [B, T]
        rewards_batch = rewards_batch.reshape(-1).double()   # [B * T]
        
        # === optimize actor for K epochs===
        loss_epochs = 0
        full_batch_size = old_ope_ma_adj.size(0)
        num_complete_minibatches = math.floor(full_batch_size / minibatch_size)
        for _ in range(K_epochs):
            for i in range(num_complete_minibatches+1):
                if i < num_complete_minibatches:
                    start_idx = i * minibatch_size
                    end_idx = (i + 1) * minibatch_size
                else:
                    start_idx = i * minibatch_size
                    end_idx = full_batch_size
        
                logprobs, state_values, dist_entropy = \
                    self.evaluate(
                        old_ope_ma_adj[start_idx: end_idx, :, :],
                        old_ope_pre_adj[start_idx: end_idx, :, :],
                        old_ope_sub_adj[start_idx: end_idx, :, :],
                        old_raw_opes[start_idx: end_idx, :, :], 
                        old_raw_mas[start_idx: end_idx, :, :], 
                        old_proc_time[start_idx: end_idx, :, :], 
                        old_ope_step_batch[start_idx: end_idx, :], 
                        old_eligible[start_idx: end_idx, :], 
                        old_action_indexes[start_idx: end_idx]
                        )   # [mini_B, ] | [mini_B, ] | [mini_B, ]
                ratios = torch.exp(logprobs - old_logprobs[i*minibatch_size:(i+1)*minibatch_size].detach())
                advantages = rewards_batch[i*minibatch_size:(i+1)*minibatch_size] - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
                loss = - A_coeff * torch.min(surr1, surr2)\
                    + vf_coeff * self.MseLoss(state_values, rewards_batch[i*minibatch_size:(i+1)*minibatch_size])\
                    - entropy_coeff * dist_entropy
                loss_epochs += loss.mean().detach()

                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()
        

        return loss_epochs.item() / K_epochs