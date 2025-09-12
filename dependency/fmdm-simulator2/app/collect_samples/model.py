# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import clip
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.nn import functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import transformers

from .trajectory_gpt2 import GPT2Model
from .transformer_encoder import Encoder

import gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from gym.wrappers.atari_preprocessing import AtariPreprocessing

from env_utils.meta_world_utils import make_meta_world_env
from env_utils.make_envsset import make_env, make_dmc_env
from data.metrics import LogScore
from env_utils.ontology import atari_games, modular_rl_games, metaworld_games, dmc_games, babyai_games


class GeneralAgent(pl.LightningModule):
    def __init__(self, config, datamodule):
        super(GeneralAgent, self).__init__()
        self.config = config

        self.datamodule = datamodule

        self.max_length = config.context_length
        self.hidden_size = config.n_embd

        GPT_config = transformers.GPT2Config(
            vocab_size=1, n_ctx=1024, n_embd=self.hidden_size, n_layer=config.n_layer, n_head=config.n_head,
            n_inner=config.n_embd * 4, activation_function=config.activation_function, n_positions=1024,
            resid_pdrop=config.resid_pdrop, attn_pdrop=config.attn_pdrop
        )

        self.token_size = config.rl_vocab_size
        self.atari_vocab_size = config.atari_vocab_size
        self.babyai_vocab_size = config.babyai_vocab_size
        self.discret_vocab_size = config.discret_vocab_size
        self.continous_vocab_size = config.continous_vocab_size
        self.token_interval = 1
        self.max_ep_len = 1000

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(GPT_config)
        # change to parallelize mode for metaworld big model
        # self.transformer.parallelize()

        self.local_embed_timestep = nn.Embedding(512, self.hidden_size)
        self.prompt_local_embed_timestep = nn.Embedding(512, self.hidden_size)

        self.encoder_state = Encoder(src_vocab_size=self.token_size, d_model=self.hidden_size, n_heads=3, d_k=64,
                                     d_v=64, d_ff=self.hidden_size, n_layers=3)
        self.prompt_encoder_state = Encoder(src_vocab_size=self.token_size, d_model=self.hidden_size, n_heads=3, d_k=64,
                                            d_v=64, d_ff=self.hidden_size, n_layers=3)
        self.image_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                           nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                           nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                           nn.Flatten(), nn.Linear(3136, self.hidden_size), nn.Tanh())

        self.src_emb = nn.Embedding(self.token_size, self.hidden_size)  # 把字转换字向量
        self.prompt_src_emb = nn.Embedding(self.token_size, self.hidden_size)  # 把字转换字向量

        self.embed_ln = nn.LayerNorm(self.hidden_size)

        self.predict_state = torch.nn.Linear(self.hidden_size, 1)

        self.predict_action = torch.nn.Linear(self.hidden_size, self.token_size - 2, bias=False)

        self.predict_return = torch.nn.Linear(self.hidden_size, 1)

        self.clip_model, self.preprocess = clip.load(self.config.clip_path)
        self.prompt_encoder_language = torch.nn.Linear(512, self.hidden_size, bias=False)

        self.metric = LogScore(config)  # used for record the data of all gpu
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(self, states, actions, text, lcoal_timesteps, env_name, attention_mask=None, prompt=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        act_dim = actions.view(batch_size, seq_length, -1).shape[
            2]  # when evaluation the act_dim is not equal max-act_dim+1

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)  # batchsize, seq_len,

        state_embeddings = torch.zeros(batch_size, seq_length, self.hidden_size, device=states.device).unsqueeze(2)
        for i in range(batch_size):
            if env_name[i] in modular_rl_games + metaworld_games:
                s_embeddings, _ = self.encoder_state(states[i][states[i] > 0].view(1 * seq_length, -1))
                state_embeddings[i] = s_embeddings.view(1, seq_length, -1).unsqueeze(2)
            elif env_name[i] in babyai_games:
                s_embeddings, _ = self.encoder_state(
                    states[i][:, :148].view(1 * seq_length, -1))  # 148=7*7*7(image)+1(direction)
                state_embeddings[i] = s_embeddings.view(1, seq_length, -1).unsqueeze(2)
            elif env_name[i] in atari_games + dmc_games:
                s = states[i] / 255.
                s_embeddings = self.image_encoder(s.reshape(-1, 4, 84, 84).type(torch.float32).contiguous())
                state_embeddings[i] = s_embeddings.reshape(1, seq_length, -1).unsqueeze(2)

        sa_embeddings = torch.zeros(batch_size, seq_length * (1 + act_dim), self.hidden_size, device=actions.device)
        for i in range(batch_size):
            if env_name[i] in babyai_games:
                action_embeddings = self.src_emb(actions[i]).view(seq_length, -1, self.hidden_size)
            else:
                action_embeddings = self.src_emb(actions[i][actions[i] > 0]).view(seq_length, -1, self.hidden_size)
            padd_embeddings = self.src_emb(
                torch.zeros(seq_length, act_dim - action_embeddings.size(1), device=actions.device).long()).view(-1,
                                                                                                                 self.hidden_size)
            sa = torch.cat((state_embeddings[i], action_embeddings), dim=1).view(-1, self.hidden_size)
            sa_embeddings[i] = torch.cat((sa, padd_embeddings), dim=0)

        sa_embeddings = sa_embeddings.view(batch_size, -1, self.hidden_size)
        local_time_embeddings = self.local_embed_timestep(lcoal_timesteps)  # # batch_size, seq_len,dim
        sa_embeddings = sa_embeddings + local_time_embeddings

        stacked_inputs = self.embed_ln(sa_embeddings)
        stacked_attention_mask = attention_mask

        # process prompt the same as d-t
        if prompt is not None:

            prompt_states, prompt_actions, prompt_text, prompt_attention_mask, prompt_local_timesteps = prompt
            prompt_seq_length = prompt_states.shape[1]  # shape batchsize, pro_len,dim

            prompt_state_embeddings = torch.zeros(batch_size, prompt_seq_length, self.hidden_size,
                                                  device=states.device).unsqueeze(2)

            prompt_language_embeddings = torch.zeros(batch_size, 1, self.hidden_size, device=states.device)
            language_embeddings = torch.zeros(batch_size, 1, self.hidden_size, device=states.device)

            for i in range(batch_size):
                if env_name[i] in modular_rl_games + metaworld_games:
                    prompt_s_embeddings, _ = self.prompt_encoder_state(
                        prompt_states[i][prompt_states[i] > 0].view(1 * prompt_seq_length, -1))
                    prompt_state_embeddings[i] = prompt_s_embeddings.view(1, prompt_seq_length, -1).unsqueeze(2)
                elif env_name[i] in babyai_games:  # 148=7*7*7(image)+1(direction)
                    prompt_s_embeddings, _ = self.prompt_encoder_state(
                        prompt_states[i][:, :148].view(1 * prompt_seq_length, -1))
                    prompt_state_embeddings[i] = prompt_s_embeddings.view(1, prompt_seq_length, -1).unsqueeze(2)
                    prompt_language_token = clip.tokenize(prompt_text[i]).to(states.device)  # source language
                    language_token = clip.tokenize(text[i]).to(states.device)  # target language
                    with torch.no_grad():
                        prompt_l_embeddings = self.clip_model.encode_text(prompt_language_token)
                        l_embeddings = self.clip_model.encode_text(language_token)
                    prompt_language_embeddings[i] = self.prompt_encoder_language(
                        prompt_l_embeddings.type(torch.float32))
                    language_embeddings[i] = self.prompt_encoder_language(l_embeddings.type(torch.float32))
                elif env_name[i] in atari_games + dmc_games:
                    s = prompt_states[i] / 255.
                    s_embeddings = self.image_encoder(s.reshape(-1, 4, 84, 84).type(torch.float32).contiguous())
                    prompt_state_embeddings[i] = s_embeddings.reshape(1, prompt_states.shape[1],
                                                                      self.config.n_embd).unsqueeze(2)

            prompt_sa_embeddings = torch.zeros(batch_size, prompt_seq_length * (2 + self.config.max_action_dim) + 2,
                                               self.hidden_size,
                                               device=actions.device)  ## batch_size, prompt_seq_len*(token_len), dim
            for i in range(batch_size):
                if env_name[i] in babyai_games:
                    prompt_action_embeddings = self.prompt_src_emb(prompt_actions[i]).view(prompt_seq_length, -1,
                                                                                           self.hidden_size)
                else:
                    prompt_action_embeddings = self.prompt_src_emb(prompt_actions[i][prompt_actions[i] > 0]).view(
                        prompt_seq_length, -1, self.hidden_size)
                prompt_padd_embeddings = self.src_emb(
                    torch.zeros(prompt_seq_length, self.config.max_action_dim + 1 - prompt_action_embeddings.size(1),
                                device=actions.device).long()).view(-1, self.hidden_size)
                prompt_sa = torch.cat((prompt_state_embeddings[i], prompt_action_embeddings), dim=1).view(-1,
                                                                                                          self.hidden_size)
                prompt_sa_embeddings[i] = torch.cat(
                    (prompt_padd_embeddings, prompt_language_embeddings[i], prompt_sa, language_embeddings[i]), dim=0)

            prompt_sa_embeddings = prompt_sa_embeddings.view(batch_size, -1, self.hidden_size)
            prompt_local_time_embeddings = self.prompt_local_embed_timestep(
                prompt_local_timesteps)  # batch_size, prompt_seq_len*(token),
            prompt_stacked_inputs = prompt_sa_embeddings + prompt_local_time_embeddings

            if prompt_stacked_inputs.shape[0] != stacked_inputs.shape[0]:
                batchsize = min(prompt_stacked_inputs.shape[0], stacked_inputs.shape[0])
                stacked_inputs = torch.cat(
                    (prompt_stacked_inputs[-batchsize:, :, :], stacked_inputs[-batchsize:, :, :]), dim=1)
                stacked_attention_mask = torch.cat(
                    (prompt_attention_mask[-batchsize:, :], stacked_attention_mask[-batchsize:, :]), dim=1)
            else:
                stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_attention_mask, stacked_attention_mask), dim=1)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']  # batchsize,3,sequence_length+prompt_length,dim

        if prompt is not None:
            return_preds = self.predict_return(x)[:, -(stacked_inputs.size(-2) - prompt_stacked_inputs.size(-2)):, :]
            state_preds = self.predict_state(x)[:, -(stacked_inputs.size(-2) - prompt_stacked_inputs.size(-2)):, :]
            action_preds = self.predict_action(x)[:, -(stacked_inputs.size(-2) - prompt_stacked_inputs.size(-2)):, :]
        else:
            return_preds = self.predict_return(x)[:, -(stacked_inputs.size(-2)):,
                           :]  # predict next return given state and action
            state_preds = self.predict_state(x)[:, -(stacked_inputs.size(-2)):,
                          :]  # predict next state given state and action
            action_preds = self.predict_action(x)[:, -(stacked_inputs.size(-2)):, :]  # predict next action given state

        return state_preds, action_preds, return_preds

    def mu_law(self, x):
        u = 100
        M = 256
        f = torch.sign(x) * torch.log(torch.abs(x) * u + 1.0) / torch.log(torch.tensor(M * u + 1.0))
        return f

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        for name, param in self.named_parameters():
            if "clip_model" in name:
                param.requires_grad = False
        critic_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(critic_params, lr=self.config.learning_rate,
                                      weight_decay=self.config.weight_decay)
        return optimizer

    # learning rate warm-up
    def optimizer_step(self, epoch,
                       batch_idx,
                       optimizer,
                       optimizer_idx,
                       optimizer_closure,
                       on_tpu,
                       using_native_amp,
                       using_lbfgs,
                       ):
        # update params
        optimizer.step(closure=optimizer_closure)
        # manually warm up lr without a scheduler
        # if self.config.wa

        if self.config.warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.config.warmup_steps)
            # lr_scale = min(1.0, float(self.trainer.global_step + 1) / (2*self.config.val_interval))

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.config.learning_rate

    def training_step(self, batch, batch_idx):

        state, action_fwd, text, fwd_mask, loss_mask, label, local_timestep, prompt_state, prompt_action_fwd, prompt_text, prompt_fwd_mask, prompt_local_timestep, env_name = batch
        prompt_batch = prompt_state, prompt_action_fwd, prompt_text, prompt_fwd_mask, prompt_local_timestep
        _, action_preds, _ = self(state, action_fwd, text, local_timestep, env_name, fwd_mask, prompt=prompt_batch)
        loss = None
        if label is not None:
            batch_size, token_seq, action_token = action_preds.shape[0], action_preds.shape[1], action_preds.shape[2]
            action_preds = action_preds.reshape(batch_size * token_seq, action_token)  #

            mask_preds = loss_mask * fwd_mask  # batch_size, seq_len,
            action_target = label[mask_preds.bool()]
            action_preds = action_preds.masked_select(
                mask_preds.bool().view(-1).unsqueeze(1).expand(-1, action_token)).contiguous().view(-1, action_token)
            loss = F.cross_entropy(action_preds, action_target)
        tb = self.logger.experiment
        self.log('train_loss', loss, on_step=True, sync_dist=True)
        tb.add_scalar('train_loss_epoch', loss, global_step=int(self.global_step / self.config.val_interval))

        return loss

    def test_step(self, batch, batch_idx):

        prompt_state, prompt_action_fwd, prompt_fwd_mask, prompt_global_timestep, prompt_local_timestep, env_name = batch

        if env_name[1:] == env_name[:-1]:
            env = self.config.env_list[self.config.game.index(env_name[0])]  # get the env_name index in the env_list
            state_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
            state = env.reset()
            num_envs = state.shape[0]
            states = torch.from_numpy(state).reshape(num_envs, 1, state_dim).to(device=prompt_state.device,
                                                                                dtype=torch.float32)
            actions = torch.zeros((num_envs, 0, act_dim), device=prompt_action_fwd.device, dtype=torch.float32)
            timesteps = torch.tensor(0, device=prompt_state.device, dtype=torch.long).repeat(num_envs).reshape(num_envs,
                                                                                                               1)

            dones = torch.zeros(num_envs, device=prompt_state.device) > 1

            episode_return, episode_length = torch.zeros(num_envs, device=prompt_state.device), torch.zeros(num_envs,
                                                                                                            device=prompt_state.device)

            for t in range(self.config.env_info[env_name[0]]['max_ep_len']):
                actions = torch.cat([actions, torch.zeros((num_envs, 1, act_dim), device=prompt_action_fwd.device)],
                                    dim=1)
                action = self.get_action(num_envs, states, actions, timesteps, state_dim, act_dim, batch[:-1])

                actions[:, -1, :] = action
                action = action.detach().cpu().numpy()

                state, reward, done, infos = env.step(action)

                cur_state = torch.from_numpy(state).to(device=prompt_state.device).reshape(num_envs, 1, state_dim)
                states = torch.cat([states, cur_state], dim=1)

                timesteps = torch.cat(
                    [timesteps, torch.ones((num_envs, 1), device=prompt_state.device, dtype=torch.long) * (t + 1)],
                    dim=1)

                episode_return += torch.from_numpy(reward).to(device=prompt_state.device) * ~dones
                episode_length += 1 * ~dones

                if done.any():
                    dones[torch.where(torch.from_numpy(done).to(device=prompt_state.device))] = True

                if dones.all():
                    break
            expert_mean = self.config.env_info[env_name[0]]['expert_mean']

    def pad_dmc_obs(self, x):
        ''' x: DMC image, shape = (H, W, C)
        '''
        x = x[np.newaxis, ...]  # N, H, W, C
        x = np.transpose(x, [0, 3, 1, 2])
        y = np.zeros((1, 4, 84, 84))
        y[:, :3, 10:10 + 64, 10:10 + 64] = x
        return y

    def validation_step(self, batch, batch_idx):
        prompt_state, prompt_action_fwd, prompt_text, prompt_fwd_mask, prompt_local_timestep, env_name = batch
        prompt_batch = prompt_state, prompt_action_fwd, prompt_text, prompt_fwd_mask, prompt_local_timestep
        episode_return, episode_length = torch.zeros(10, device=prompt_state.device), torch.zeros(10,
                                                                                                  device=prompt_state.device)
        if env_name[1:] == env_name[:-1]:
            if env_name[0] in metaworld_games:
                env_id = metaworld_games.index(env_name[0])
                env = SubprocVecEnv([make_meta_world_env(env_id, i) for i in range(self.config.num_eval_episodes)])
                env_type = torch.ones(1, device=prompt_state.device) * 3

                state_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
                state = env.reset()
                num_envs = state.shape[0]
                states = torch.from_numpy(state).reshape(num_envs, 1, state_dim).to(device=prompt_state.device,
                                                                                    dtype=torch.float32)
                actions = torch.zeros((num_envs, 0, act_dim), device=prompt_action_fwd.device, dtype=torch.float32)
                timesteps = torch.tensor(0, device=prompt_state.device, dtype=torch.long).repeat(num_envs).reshape(
                    num_envs, 1)

                dones = torch.zeros(num_envs, device=prompt_state.device) > 1
                successes = torch.zeros(num_envs, device=prompt_state.device)

                episode_return, episode_length = torch.zeros(num_envs, device=prompt_state.device), torch.zeros(
                    num_envs, device=prompt_state.device)

                for t in range(self.config.env_info[env_name[0]]['max_ep_len']):
                    actions = torch.cat([actions, torch.zeros((num_envs, 1, act_dim), device=prompt_action_fwd.device)],
                                        dim=1)
                    action = self.get_action(num_envs, states, actions, timesteps, state_dim, act_dim, prompt_batch,
                                             env_name, env_type)

                    actions[:, -1, :] = action
                    action = action.detach().cpu().numpy()

                    state, reward, done, infos = env.step(action)

                    cur_state = torch.from_numpy(state).to(device=prompt_state.device).reshape(num_envs, 1, state_dim)
                    states = torch.cat([states, cur_state], dim=1)

                    timesteps = torch.cat(
                        [timesteps, torch.ones((num_envs, 1), device=prompt_state.device, dtype=torch.long) * (t + 1)],
                        dim=1)

                    episode_return += torch.from_numpy(reward).to(device=prompt_state.device) * ~dones
                    episode_length += 1 * ~dones

                    if done.any():
                        dones[torch.where(torch.from_numpy(done).to(device=prompt_state.device))] = True

                    if dones.all():
                        break
                expert_mean = self.config.env_info[env_name[0]]['expert_mean']
        #     if env_name[0] in atari_games:
        #         env = gym.make(env_name[0], full_action_space=True)
        #         env = AtariPreprocessing(
        #                 env, noop_max=30, frame_skip=1, screen_size=84,  # frame_skip=1
        #                 grayscale_obs=True, scale_obs=False)
        #         # env = VecFrameStack(env, n_stack=4)
        #         state_dim, act_dim = 4*84*84, 1
        #         if int(self.global_step/self.config.val_interval) == 0:
        #             ep_len = 10
        #         else:
        #             ep_len = 108000
        #         env_type = torch.ones(1,device=prompt_state.device)

        #         episode_return, episode_length = torch.zeros(10,device=prompt_state.device), torch.zeros(10,device=prompt_state.device)
        #         for i in range(10):
        #             state = env.reset()
        #             state = ((state,)*4)

        #             states = torch.tensor(np.array(state)).permute(1,2,0).reshape(1,1,state_dim).to(device=prompt_state.device, dtype=torch.float32)  # 4 84 84 -- 84 84 4 -- 1 1 28224

        #             num_envs = states.shape[0]

        #             actions = torch.zeros((num_envs, 0, act_dim), device=prompt_action_fwd.device, dtype=torch.int)

        #             timesteps = torch.tensor(0,device=prompt_state.device,dtype=torch.long).repeat(num_envs).reshape(num_envs, 1)

        #             dones = False

        #             for t in range(ep_len):
        #                 actions = torch.cat([actions, torch.zeros((num_envs, 1, act_dim), device=prompt_action_fwd.device)], dim=1)
        #                 action = self.get_action(num_envs, states, actions, timesteps, state_dim, act_dim, prompt_batch, env_name, env_type)
        #                 actions[:, -1, :] = action
        #                 action = action.detach().cpu().numpy()

        #                 for k in range(4):
        #                     state_, reward, done, info = env.step(action)
        #                     episode_return[i] += reward
        #                     state_ = torch.tensor(state_, dtype=torch.float32).type(torch.float32).unsqueeze(0)
        #                     if k == 0:
        #                         state = state_
        #                     else:
        #                         state = torch.cat([state, state_], dim=0)

        #                 cur_state = state.permute(1,2,0).to(device=prompt_state.device).reshape(num_envs, 1, state_dim)
        #                 states = torch.cat([states, cur_state], dim=1)

        #                 timesteps = torch.cat([timesteps,torch.ones((num_envs, 1), device=prompt_state.device, dtype=torch.long) * (t + 1)], dim=1)

        #                 episode_length[i] += 1

        #                 if done:
        #                     break

        #     elif env_name[0] in modular_rl_games:
        #         env_id = "%s-v0" % env_name[0]
        #         env = SubprocVecEnv([make_env(env_id, i) for i in range(self.config.num_eval_episodes)])
        #         state_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
        #         ep_len = self.config.env_info[env_name[0]]['max_ep_len']
        #         env_type = torch.ones(1,device=prompt_state.device) * 2

        #         state = env.reset()

        #         num_envs = state.shape[0]
        #         states = torch.from_numpy(state).reshape(num_envs,1,state_dim).to(device=prompt_state.device, dtype=torch.float32)
        #         actions = torch.zeros((num_envs, 0, act_dim), device=prompt_action_fwd.device, dtype=torch.float32)

        #         timesteps = torch.tensor(0,device=prompt_state.device,dtype=torch.long).repeat(num_envs).reshape(num_envs, 1)

        #         dones = torch.zeros(num_envs,device=prompt_state.device) > 1

        #         episode_return, episode_length = torch.zeros(num_envs,device=prompt_state.device), torch.zeros(num_envs,device=prompt_state.device)

        #         for t in range(ep_len):
        #             actions = torch.cat([actions, torch.zeros((num_envs, 1, act_dim), device=prompt_action_fwd.device)], dim=1)
        #             action = self.get_action(num_envs, states, actions, timesteps, state_dim, act_dim, prompt_batch, env_name,
        #                 env_type)
        #             actions[:, -1, :] = action
        #             action = action.detach().cpu().numpy()

        #             state, reward, done, infos = env.step(action)

        #             cur_state = torch.from_numpy(state).to(device=prompt_state.device).reshape(num_envs, 1, state_dim)
        #             states = torch.cat([states, cur_state], dim=1)

        #             timesteps = torch.cat([timesteps,torch.ones((num_envs, 1), device=prompt_state.device, dtype=torch.long) * (t + 1)], dim=1)

        #             episode_return += torch.from_numpy(reward).to(device=prompt_state.device) * ~dones
        #             episode_length += 1 * ~dones

        #             if done.any():
        #                 dones[torch.where(torch.from_numpy(done).to(device=prompt_state.device))] = True

        #             if dones.all():
        #                 break
        #         env.close()
        #     elif env_name[0] in metaworld_games:
        #         env_id = metaworld_games.index(env_name[0])
        #         env = SubprocVecEnv([make_meta_world_env(env_id, i) for i in range(self.config.num_eval_episodes)])
        #         env_type = torch.ones(1,device=prompt_state.device) * 5

        #         state_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
        #         state = env.reset()
        #         num_envs = state.shape[0]
        #         states = torch.from_numpy(state).reshape(num_envs,1,state_dim).to(device=prompt_state.device, dtype=torch.float32)
        #         actions = torch.zeros((num_envs, 0, act_dim), device=prompt_action_fwd.device, dtype=torch.float32)
        #         timesteps = torch.tensor(0, device=prompt_state.device, dtype=torch.long).repeat(num_envs).reshape(num_envs, 1)

        #         dones = torch.zeros(num_envs,device=prompt_state.device) > 1
        #         successes = torch.zeros(num_envs,device=prompt_state.device)

        #         episode_return, episode_length = torch.zeros(num_envs,device=prompt_state.device), torch.zeros(num_envs,device=prompt_state.device)

        #         for t in range(self.config.env_info[env_name[0]]['max_ep_len']):
        #             actions = torch.cat([actions, torch.zeros((num_envs, 1, act_dim), device=prompt_action_fwd.device)], dim=1)
        #             action = self.get_action( num_envs, states, actions, timesteps, state_dim, act_dim, batch[:-1], env_name, env_type)

        #             actions[:, -1, :] = action
        #             action = action.detach().cpu().numpy()

        #             state, reward, done, infos = env.step(action)
        #             cur_state = torch.from_numpy(state).to(device=prompt_state.device).reshape(num_envs, 1, state_dim)
        #             states = torch.cat([states, cur_state], dim=1)
        #             # rewards[:, -1] = torch.from_numpy(reward).to(device=device)
        #             timesteps = torch.cat([timesteps,torch.ones((num_envs, 1), device=prompt_state.device, dtype=torch.long) * (t + 1)], dim=1)

        #             episode_return += torch.from_numpy(reward).to(device=prompt_state.device) * ~dones
        #             episode_length += 1 * ~dones

        #             if done.any():
        #                 dones[torch.where(torch.from_numpy(done).to(device=prompt_state.device))] = True

        #             success = np.array([info["success"] for info in infos])
        #             if success.any():
        #                 successes[torch.where(torch.from_numpy(success).to(device=prompt_state.device))] = 1.

        #             if dones.all():
        #                 break
        #         env.close()
        #         print("==================================================")
        #         print(f"Metaworld success rate: {successes.mean()}")
        #     elif env_name[0] in dmc_games:
        #         env = make_dmc_env(env_name[0], 0)
        #         state_dim = 4*84*84
        #         act_dim = env.action_space.shape[0]
        #         ep_len = self.config.env_info[env_name[0]]['max_ep_len']

        #         state_ = env.reset()
        #         state_ = state_['image']
        #         state = self.pad_dmc_obs(state_.copy())

        #         env_type = torch.ones(1,device=prompt_state.device)
        #         num_envs = 1
        #         states = torch.from_numpy(state.copy()).reshape(num_envs, 1, state_dim).to(device=prompt_state.device, dtype=torch.float32)
        #         actions = torch.zeros((num_envs, 0, act_dim), device=prompt_action_fwd.device, dtype=torch.float32)
        #         timesteps = torch.tensor(0,device=prompt_state.device,dtype=torch.long).repeat(num_envs).reshape(num_envs, 1)
        #         dones = torch.zeros(num_envs,device=prompt_state.device) > 1

        #         episode_return, episode_length = torch.zeros(num_envs, device=prompt_state.device), torch.zeros(num_envs, device=prompt_state.device)
        #         for i in range(ep_len):
        #             actions = torch.cat([actions, torch.zeros((num_envs, 1, act_dim), device=prompt_action_fwd.device)], dim=1)
        #             action = self.get_action(num_envs, states, actions, timesteps, state_dim, act_dim, prompt_batch, env_name, env_type)
        #             actions[:, -1, :] = action
        #             action = action.detach().cpu().numpy()

        #             state_, reward, done, infos = env.step(action)
        #             state_ = state_['image']
        #             state = self.pad_dmc_obs(state_)

        #             cur_state = torch.from_numpy(state.copy()).to(device=prompt_state.device).reshape(num_envs, 1, state_dim)
        #             states = torch.cat([states, cur_state], dim=1)

        #             timesteps = torch.cat([timesteps,torch.ones((num_envs, 1), device=prompt_state.device, dtype=torch.long) * (i + 1)], dim=1)

        #             episode_return += (torch.tensor(reward).to(device=prompt_state.device) * ~done)
        #             episode_length += (1 * ~done)

        #             if done:
        #                 dones[torch.where(torch.tensor(done).to(device=prompt_state.device))] = True

        #             if done:
        #                 break
        #         env.close()
        #         print('='*30)
        #         print(f'DMC Testing results\n{env_name[0]}: ep_return: {episode_return.mean()}')
        #     elif env_name[0] in babyai_games:
        #         import pickle as pkl
        #         with open('data/vocab.pkl', 'rb') as f:
        #             vocab = pkl.load(f)
        #         def encode(s):
        #             lis = s.split()
        #             for i in range(len(lis)):
        #                 lis[i]=vocab[lis[i]]
        #             return lis
        #         MAX_LENGTH=11

        #         task_name = env_name[0].split('-')[-1]
        #         env = gym.make('BabyAI-'+task_name+'-v0')
        #         state_dim, act_dim = 162, 1
        #         ep_len = self.config.env_info[env_name[0]]['max_ep_len']
        #         env_type = torch.ones(1,device=prompt_state.device)*4

        #         state = env.reset()
        #         task_encode = np.array(encode(state['mission'])).reshape(1,-1)
        #         if task_encode.shape[1]>MAX_LENGTH:
        #             return
        #         left_length = (MAX_LENGTH-task_encode.shape[1])//2
        #         right_length = MAX_LENGTH-left_length-task_encode.shape[1]
        #         task_encode = np.pad(task_encode, ((0,0), (left_length,right_length)), 'constant', constant_values=(0,0))
        #         observations = state['image'].reshape(-1,147)
        #         directions = np.eye(4)[state['direction']].astype(np.uint8).reshape(-1,4)
        #         state = np.concatenate((task_encode, observations, directions), axis=1)

        #         num_envs = state.shape[0]
        #         states = torch.from_numpy(state).reshape(num_envs,1,state_dim).to(device=prompt_state.device, dtype=torch.float32)
        #         # states += 27 # move to discrete vocab space, +27 for [1, 18] -> Atari and [18, 26]: babyai action
        #         actions = torch.zeros((num_envs, 0, act_dim), device=prompt_action_fwd.device, dtype=torch.float32)

        #         timesteps = torch.tensor(0,device=prompt_state.device,dtype=torch.long).repeat(num_envs).reshape(num_envs, 1)

        #         episode_return, episode_length = torch.zeros(num_envs,device=prompt_state.device), torch.zeros(num_envs,device=prompt_state.device)

        #         for t in range(ep_len):
        #             actions = torch.cat([actions, torch.zeros((num_envs, 1, act_dim), device=prompt_action_fwd.device)], dim=1)
        #             action = self.get_action(num_envs, states, actions, timesteps, state_dim, act_dim, prompt_batch, env_name, env_type)
        #             actions[:, -1, :] = action
        #             action = action.detach().cpu().numpy()

        #             state, reward, done, infos = env.step(action)
        #             task_encode = np.array(encode(state['mission'])).reshape(1,-1)
        #             left_length = (MAX_LENGTH-task_encode.shape[1])//2
        #             right_length = MAX_LENGTH-left_length-task_encode.shape[1]
        #             task_encode = np.pad(task_encode,((0,0),(left_length,right_length)),'constant',constant_values=(0,0))
        #             observations = state['image'].reshape(-1,147)
        #             directions = np.eye(4)[state['direction']].astype(np.uint8).reshape(-1,4)
        #             state = np.concatenate((task_encode,observations,directions),axis=1)

        #             cur_state = torch.from_numpy(state).to(device=prompt_state.device).reshape(num_envs, 1, state_dim)
        #             states = torch.cat([states, cur_state], dim=1)

        #             timesteps = torch.cat([timesteps,torch.ones((num_envs, 1), device=prompt_state.device, dtype=torch.long) * (t + 1)], dim=1)

        #             episode_return += (torch.tensor(reward).to(device=prompt_state.device) * ~done)
        #             episode_length += (1 * ~done)

        #             if done:
        #                 break
        #         env.close()
        #         print('='*30)
        #         print(f'=======Babyai testing results {env_name[0]}: ep_return = {episode_return.mean()}, episode_length = {episode_length.mean()}')
        log_dict = {'env': env_name[0],
                    'expert': round(self.config.env_info[env_name[0]]['expert_mean'], 2),
                    'score': episode_return.mean() / self.config.env_info[env_name[0]]['expert_mean'] * 100,
                    'mean': episode_return.mean(),
                    'std': episode_return.std()}
        print('####', int(self.global_step / self.config.val_interval), self.global_step, log_dict,
              " average episode_length: ", episode_length.mean())
        self.metric.update(log_dict)

    def validation_epoch_end(self, validation_step_outs):

        results, sample_factor = self.metric.compute()
        sample_prob = torch.tensor(sample_factor) / torch.tensor(sample_factor).sum()

        np.save(
            '../model_saved/' + os.path.basename(os.getcwd()) + '/sample_prob/' + self.config.sample_prob_name + '.npy',
            sample_prob)
        tb = self.logger.experiment
        # epoch = (int(1e+6*len(self.config.game)/self.config.batch_size/self.trainer.devices))
        # config.val_interval
        for res in results:
            # self.log(res, results[res])
            tb.add_scalar(res, results[res], global_step=int(self.global_step))
            tb.add_scalar(res + '_epoch', results[res], global_step=int(self.global_step / self.config.val_interval))
            # tb.add_
        self.metric.reset()
        if self.global_rank == 0:
            print(f'===save_dir==={self.logger.save_dir}/checkpoint/{self.config.labels}/', )

    def get_action(self, num_envs, states, actions, timesteps, state_dim, act_dim, prompt, env_name, env_name_type):
        # we don't care about the past rewards in this model
        states = states.reshape(num_envs, -1, state_dim)
        actions = actions.reshape(num_envs, -1, act_dim)
        timesteps = timesteps.reshape(num_envs, -1)

        if self.max_length is not None:  # padding with zero, or cutimestepst with max_length
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])]).repeat(num_envs)
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(num_envs, -1)

            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length - states.shape[1], state_dim), device=states.device),
                 states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], act_dim), device=actions.device),
                 actions],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device),
                 timesteps],
                dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        s_b_size, seq_len, s_dim = states.size()

        if env_name[0] in modular_rl_games:
            states = torch.clip(self.mu_law(states), -1, 1)
            states = ((1 + states) / 2 * (self.continous_vocab_size - 1) + self.discret_vocab_size + 1).long()
            actions = torch.clip(actions, -1, 1)
            actions = ((1 + actions) / 2 * (self.continous_vocab_size - 1) + self.discret_vocab_size + 1).long()
        elif env_name[0] in metaworld_games:
            # add clip
            states = torch.clip(self.mu_law(states), -1, 1)
            states = ((1 + states) / 2 * (self.continous_vocab_size - 1) + self.discret_vocab_size + 1).long()
            actions = torch.clip(self.mu_law(actions), -1, 1)
            actions = ((1 + actions) / 2 * (self.continous_vocab_size - 1) + self.discret_vocab_size + 1).long()
        elif env_name[0] in atari_games:
            states = states.long()
            actions = (actions + 1).long()
        elif env_name[0] in dmc_games:
            states = states.long()
            actions = torch.clip(actions, -1, 1)
            actions = ((1 + actions) / 2 * (self.continous_vocab_size - 1) + self.discret_vocab_size + 1).long()
        elif env_name[0] in babyai_games:
            states = states.long()
            actions = (actions + 19).long()

        actions_start = (torch.ones(actions.size(0), actions.size(1), 1) * (self.token_size - 2)).to(dtype=torch.long,
                                                                                                     device=states.device)
        actions = torch.cat((actions_start, actions), dim=2)

        mask = attention_mask.repeat(1, 1 + actions.size(-1)).view(states.size(0), -1, self.max_length).transpose(2,
                                                                                                                  1).reshape(
            states.size(0), (1 + actions.size(-1)) * self.max_length)  # 48,5,17+7

        state_timestep = torch.arange(1, 1 + 1).repeat(s_b_size * seq_len).view(s_b_size, seq_len, 1).to(
            dtype=torch.long, device=states.device)
        action_timestep = torch.ones(s_b_size, seq_len, actions.size(-1)).to(dtype=torch.long, device=states.device) * (
                    1 + 1)
        local_timestep = torch.cat((state_timestep, action_timestep), dim=2).view(s_b_size, -1)

        for i in range(act_dim + 1):  # predict  a1,a2,an, end sign
            # Note: prompt within kwargs
            _, action_preds, _ = self(
                states, actions.view(s_b_size, -1), local_timestep, env_name, env_name_type, attention_mask=mask,
                prompt=prompt)
            if i < act_dim:  # don't predict the eng sign
                out_mask = torch.zeros_like(action_preds)
                out_mask[:, :, 0] = 1
                if act_dim == 17:
                    start_idx = int(
                        ((1 - 0.4) / 2 * (self.continous_vocab_size - 1) + self.discret_vocab_size + 1))  # 1331
                    end_idx = int(
                        ((1 + 0.4) / 2 * (self.continous_vocab_size - 1) + self.discret_vocab_size + 1))  # 1741
                    out_mask[:, :, :start_idx + 1] = 1
                    out_mask[:, :, end_idx + 1:] = 1
                if env_name[0] in atari_games:
                    out_mask[:, :, self.atari_vocab_size + 1:] = 1
                elif env_name[0] in babyai_games:
                    out_mask[:, :, :self.atari_vocab_size + 1] = 1
                    out_mask[:, :, self.babyai_vocab_size + 1:] = 1
                elif env_name[0] in modular_rl_games + metaworld_games + dmc_games:
                    out_mask[:, :, :self.discret_vocab_size + 1] = 1
                action_preds.masked_fill_(out_mask.bool(), -1e9)
                actions[:, -1, -act_dim + i] = action_preds[:, -act_dim - 1 + i].max(dim=-1, keepdim=False)[1]

        if env_name[0] in modular_rl_games + metaworld_games + dmc_games:
            return (actions[:, -1, -act_dim - 1:][:, 1:act_dim + 1] - 1 - self.discret_vocab_size) / (
                        self.continous_vocab_size - 1) * 2 - 1  # filter the start sign
        elif env_name[0] in atari_games:
            return actions[:, -1, -act_dim - 1:][:, 1:act_dim + 1].int() - 1
        elif env_name[0] in babyai_games:
            # move back to babyai's original action space before conducting
            return actions[:, -1, -act_dim - 1:][:, 1:act_dim + 1].int() - 19