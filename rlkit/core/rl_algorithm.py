import abc
from collections import OrderedDict
import time
import os
import glob
import gtimer as gt
import numpy as np

from rlkit.core import logger, eval_util
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.in_place import InPlacePathSampler, OfflineInPlacePathSampler
from rlkit.torch import pytorch_util as ptu

class OfflineMetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            train_tasks,
            eval_tasks,
            goal_radius,
            eval_deterministic=True,
            render=False,
            render_eval_paths=False,
            plotter=None,
            **kwargs
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval
        :param goal_radius: reward threshold for defining sparse rewards

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.env                             = env
        self.agent                           = agent
        self.train_tasks                     = train_tasks
        self.eval_tasks                      = eval_tasks
        self.goal_radius                     = goal_radius

        self.meta_batch                      = kwargs['meta_batch']
        self.batch_size                      = kwargs['batch_size']
        self.num_iterations                  = kwargs['num_iterations']
        self.num_train_steps_per_itr         = kwargs['num_train_steps_per_itr']
        self.num_initial_steps               = kwargs['num_initial_steps']
        self.num_tasks_sample                = kwargs['num_tasks_sample']
        self.num_steps_prior                 = kwargs['num_steps_prior']
        self.num_steps_posterior             = kwargs['num_steps_posterior']
        self.num_extra_rl_steps_posterior    = kwargs['num_extra_rl_steps_posterior']
        self.num_evals                       = kwargs['num_evals']
        self.num_steps_per_eval              = kwargs['num_steps_per_eval']
        self.embedding_batch_size            = kwargs['embedding_batch_size']
        self.embedding_mini_batch_size       = kwargs['embedding_mini_batch_size']
        self.max_path_length                 = kwargs['max_path_length']
        self.discount                        = kwargs['discount']
        self.replay_buffer_size              = kwargs['replay_buffer_size']
        self.reward_scale                    = kwargs['reward_scale']
        self.update_post_train               = kwargs['update_post_train']
        self.num_exp_traj_eval               = kwargs['num_exp_traj_eval']
        self.save_replay_buffer              = kwargs['save_replay_buffer']
        self.save_algorithm                  = kwargs['save_algorithm']
        self.save_environment                = kwargs['save_environment']
        self.dump_eval_paths                 = kwargs['dump_eval_paths']
        self.data_dir                        = kwargs['data_dir']
        self.train_epoch                     = kwargs['train_epoch']
        self.eval_epoch                      = kwargs['eval_epoch']
        self.sample                          = kwargs['sample']
        self.n_trj                           = kwargs['n_trj']
        self.allow_eval                      = kwargs['allow_eval']
        self.mb_replace                      = kwargs['mb_replace']

        self.eval_deterministic              = eval_deterministic
        self.render                          = render
        self.eval_statistics                 = None
        self.render_eval_paths               = render_eval_paths
        self.plotter                         = plotter
        
        self.train_buffer      = MultiTaskReplayBuffer(self.replay_buffer_size, env, self.train_tasks, self.goal_radius)
        self.eval_buffer       = MultiTaskReplayBuffer(self.replay_buffer_size, env, self.eval_tasks,  self.goal_radius)
        self.replay_buffer     = MultiTaskReplayBuffer(self.replay_buffer_size, env, self.train_tasks, self.goal_radius)
        self.enc_replay_buffer = MultiTaskReplayBuffer(self.replay_buffer_size, env, self.train_tasks, self.goal_radius)
        # offline sampler which samples from the train/eval buffer
        self.offline_sampler   = OfflineInPlacePathSampler(env=env, policy=agent, max_path_length=self.max_path_length)
        # online sampler for evaluation (if collect on-policy context, for offline context, use self.offline_sampler)
        self.sampler           = InPlacePathSampler(env=env, policy=agent, max_path_length=self.max_path_length)

        self._n_env_steps_total     = 0
        self._n_train_steps_total   = 0
        self._n_rollouts_total      = 0
        self._do_train_time         = 0
        self._epoch_start_time      = None
        self._algo_start_time       = None
        self._old_table_keys        = None
        self._current_path_builder  = PathBuilder()
        self._exploration_paths     = []
        self.init_buffer()

    def init_buffer(self):
        train_trj_paths = []
        eval_trj_paths = []
        # trj entry format: [obs, action, reward, new_obs]
        if self.sample:
            for n in range(self.n_trj):
                if self.train_epoch is None:
                    train_trj_paths += glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_evalsample%d_step*.npy" %(n)))
                else:
                    train_trj_paths += glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_evalsample%d_step%d.npy" %(n, self.train_epoch)))
                if self.eval_epoch is None:
                    eval_trj_paths += glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_evalsample%d_step*.npy" %(n)))
                else:
                    eval_trj_paths += glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_evalsample%d_step%d.npy" %(n, self.eval_epoch)))
        else:
            if self.train_epoch is None:
                train_trj_paths = glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_eval[0-%d]_step*.npy") %(self.n_trj))
            else:
                train_trj_paths = glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_eval[0-%d]_step%d.npy" %(self.n_trj, self.train_epoch)))
            if self.eval_epoch is None:
                eval_trj_paths = glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_eval[0-%d]_step*.npy") %(self.n_trj))
            else:
                eval_trj_paths = glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_eval[0-%d]_step%d.npy" %(self.n_trj, self.test_epoch)))
        
        train_paths = [train_trj_path for train_trj_path in train_trj_paths if
                       int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.train_tasks]
        train_task_idxs = [int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) for train_trj_path in train_trj_paths if
                       int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.train_tasks]
        eval_paths = [eval_trj_path for eval_trj_path in eval_trj_paths if
                      int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.eval_tasks]
        eval_task_idxs = [int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) for eval_trj_path in eval_trj_paths if
                          int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.eval_tasks]

        obs_train_lst = []
        action_train_lst = []
        reward_train_lst = []
        next_obs_train_lst = []
        terminal_train_lst = []
        task_train_lst = []
        obs_eval_lst = []
        action_eval_lst = []
        reward_eval_lst = []
        next_obs_eval_lst = []
        terminal_eval_lst = []
        task_eval_lst = []

        for train_path, train_task_idx in zip(train_paths, train_task_idxs):
            trj_npy = np.load(train_path, allow_pickle=True)
            obs_train_lst += list(trj_npy[:, 0])
            action_train_lst += list(trj_npy[:, 1])
            reward_train_lst += list(trj_npy[:, 2])
            next_obs_train_lst += list(trj_npy[:, 3])
            terminal = [0 for _ in range(trj_npy.shape[0])]
            terminal[-1] = 1
            terminal_train_lst += terminal
            task_train = [train_task_idx for _ in range(trj_npy.shape[0])]
            task_train_lst += task_train
        for eval_path, eval_task_idx in zip(eval_paths, eval_task_idxs):
            trj_npy = np.load(eval_path, allow_pickle=True)
            obs_eval_lst += list(trj_npy[:, 0])
            action_eval_lst += list(trj_npy[:, 1])
            reward_eval_lst += list(trj_npy[:, 2])
            next_obs_eval_lst += list(trj_npy[:, 3])
            terminal = [0 for _ in range(trj_npy.shape[0])]
            terminal[-1] = 1
            terminal_eval_lst += terminal
            task_eval = [eval_task_idx for _ in range(trj_npy.shape[0])]
            task_eval_lst += task_eval


        # load training buffer
        for i, (
                task_train,
                obs,
                action,
                reward,
                next_obs,
                terminal,
        ) in enumerate(zip(
            task_train_lst,
            obs_train_lst,
            action_train_lst,
            reward_train_lst,
            next_obs_train_lst,
            terminal_train_lst,
        )):
            self.train_buffer.add_sample(
                task_train,
                obs,
                action,
                reward,
                terminal,
                next_obs,
                **{'env_info': {}},
            )

        # load evaluation buffer
        for i, (
                task_eval,
                obs,
                action,
                reward,
                next_obs,
                terminal,
        ) in enumerate(zip(
            task_eval_lst,
            obs_eval_lst,
            action_eval_lst,
            reward_eval_lst,
            next_obs_eval_lst,
            terminal_eval_lst,
        )):
            self.eval_buffer.add_sample(
                task_eval,
                obs,
                action,
                reward,
                terminal,
                next_obs,
                **{'env_info': {}},
            )
        
    def _try_to_eval(self, epoch):
        #logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch)
            #params = self.get_epoch_snapshot(epoch)
            #logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys
            logger.record_tabular("Number of train steps total", self._n_train_steps_total)
            logger.record_tabular("Number of env steps total",   self._n_env_steps_total)
            logger.record_tabular("Number of rollouts total",    self._n_rollouts_total)

            times_itrs  = gt.get_times().stamps.itrs
            train_time  = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time   = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time  = train_time + sample_time + eval_time
            total_time  = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True
    
    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation,)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def _do_eval(self, indices, epoch, buffer):
        final_returns = []
        online_returns = []
        for idx in indices:
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r, buffer)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean([a[-1] for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def test(self, log_dir, end_point=-1):
        assert os.path.exists(log_dir)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(range(self.num_iterations), save_itrs=True):
            self._start_epoch(it_)
            
            if it_ == 0:
                print('collecting initial pool of data for test')
                # temp for evaluating
                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.collect_data(self.num_initial_steps, 1, np.inf, buffer=self.train_buffer)
            # Sample data from train tasks.
            for i in range(self.num_tasks_sample):
                idx = np.random.choice(self.train_tasks, 1)[0]
                self.task_idx = idx
                self.env.reset_task(idx)
                self.enc_replay_buffer.task_buffers[idx].clear()

                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    self.collect_data(self.num_steps_prior, 1, np.inf, buffer=self.train_buffer)
                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train, buffer=self.train_buffer)
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train, buffer=self.train_buffer,
                                      add_to_enc_buffer=False)
            
            print([self.replay_buffer.task_buffers[idx]._size for idx in self.train_tasks])
            print([self.enc_replay_buffer.task_buffers[idx]._size for idx in self.train_tasks])

            for train_step in range(self.num_train_steps_per_itr):
                self._n_train_steps_total += 1

            gt.stamp('train')
            # eval
            self.training_mode(False)
            if it_ % 5 == 0 and it_ > end_point:
                status = self.load_epoch_model(it_, log_dir)
                if status:
                    self._try_to_eval(it_)
            gt.stamp('eval')
            self._end_epoch()

    def train(self):
        '''
        meta-training loop
        '''
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(range(self.num_iterations), save_itrs=True):
            self._start_epoch(it_)
            self.training_mode(True)
            if it_ == 0:
                print('collecting initial pool of data for train and eval')
                # temp for evaluating
                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.collect_data(self.num_initial_steps, 1, np.inf, buffer=self.train_buffer)
            # Sample data from train tasks.
            for i in range(self.num_tasks_sample):
                idx = np.random.choice(self.train_tasks, 1)[0]
                self.task_idx = idx
                self.env.reset_task(idx)
                self.enc_replay_buffer.task_buffers[idx].clear()

                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    self.collect_data(self.num_steps_prior, 1, np.inf, buffer=self.train_buffer)
                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train, buffer=self.train_buffer)
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train, buffer=self.train_buffer,
                                      add_to_enc_buffer=False)
            
            indices_lst = []
            z_means_lst = []
            z_vars_lst = []
            # Sample train tasks and compute gradient updates on parameters.
            for train_step in range(self.num_train_steps_per_itr):
                indices = np.random.choice(self.train_tasks, self.meta_batch, replace=self.mb_replace)
                z_means, z_vars = self._do_training(indices, zloss=True)
                indices_lst.append(indices)
                z_means_lst.append(z_means)
                z_vars_lst.append(z_vars)
                self._n_train_steps_total += 1

            indices = np.concatenate(indices_lst)
            z_means = np.concatenate(z_means_lst)
            z_vars = np.concatenate(z_vars_lst)
            data_dict = self.data_dict(indices, z_means, z_vars)
            logger.save_itr_data(it_, **data_dict)
            gt.stamp('train')
            self.training_mode(False)
            # eval
            params = self.get_epoch_snapshot(it_)
            logger.save_itr_params(it_, params)

            if self.allow_eval:
                logger.save_extra_data(self.get_extra_data_to_save(it_))
                self._try_to_eval(it_)
                gt.stamp('eval')
            self._end_epoch()

    def data_dict(self, indices, z_means, z_vars):
        data_dict = {}
        data_dict['task_idx'] = indices
        for i in range(z_means.shape[1]):
            data_dict['z_means%d' %i] = list(z_means[:, i])
        for i in range(z_vars.shape[1]):
            data_dict['z_vars%d' % i] = list(z_vars[:, i])
        return data_dict

    def evaluate(self, epoch):

        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### sample trajectories from prior for debugging / visualization
        if self.dump_eval_paths:
            # 100 arbitrarily chosen for visualizations of point_robot trajectories
            # just want stochasticity of z, not the policy
            self.agent.clear_z()
            prior_paths, _ = self.offline_sampler.obtain_samples(buffer=self.train_buffer,
                                                         deterministic=self.eval_deterministic,
                                                         max_samples=self.max_path_length * 20,
                                                         accum_context=False,
                                                         resample=1)
            logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        # eval on a subset of train tasks for speed

        # {}-dir envs
        if len(self.train_tasks) == 2 and len(self.eval_tasks) == 2:
            indices = self.train_tasks
            eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
            ### eval train tasks with posterior sampled from the training replay buffer
            train_returns = []
            for idx in indices:
                self.task_idx = idx
                self.env.reset_task(idx)
                paths = []
                print(self.num_steps_per_eval, self.max_path_length)
                for _ in range(self.num_steps_per_eval // self.max_path_length):
                    context = self.sample_context(idx)
                    self.agent.infer_posterior(context, idx)
                    p, _ = self.offline_sampler.obtain_samples(buffer=self.train_buffer,
                                                               deterministic=self.eval_deterministic,
                                                               max_samples=self.max_path_length,
                                                               accum_context=False,
                                                               max_trajs=1,
                                                               resample=np.inf)
                    paths += p

                if self.sparse_rewards:
                    for p in paths:
                        sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                        p['rewards'] = sparse_rewards

                train_returns.append(eval_util.get_average_returns(paths))
            ### eval train tasks with on-policy data to match eval of test tasks
            train_final_returns, train_online_returns = self._do_eval(indices, epoch, buffer=self.train_buffer)
            eval_util.dprint('train online returns')
            eval_util.dprint(train_online_returns)

            ### test tasks
            eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
            test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch, buffer=self.eval_buffer)
            eval_util.dprint('test online returns')
            eval_util.dprint(test_online_returns)

            # save the final posterior
            self.agent.log_diagnostics(self.eval_statistics)

            if hasattr(self.env, "log_diagnostics"):
                self.env.log_diagnostics(paths, prefix=None)

            avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
            avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
            for i in indices:
                self.eval_statistics[f'AverageTrainReturn_train_task{i}'] = train_returns[i]
                self.eval_statistics[f'AverageReturn_all_train_task{i}'] = train_final_returns[i]
                self.eval_statistics[f'AverageReturn_all_test_tasks{i}'] = test_final_returns[i]

        # non {}-dir envs
        else:
            indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
            eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
            ### eval train tasks with posterior sampled from the training replay buffer
            train_returns = []
            for idx in indices:
                self.task_idx = idx
                self.env.reset_task(idx)
                paths = []
                for _ in range(self.num_steps_per_eval // self.max_path_length):
                    context = self.sample_context(idx)
                    self.agent.infer_posterior(context, idx)
                    p, _ = self.offline_sampler.obtain_samples(buffer=self.train_buffer,
                                                       deterministic=self.eval_deterministic,
                                                       max_samples=self.max_path_length,
                                                       accum_context=False,
                                                       max_trajs=1,
                                                       resample=np.inf)
                    paths += p

                if self.sparse_rewards:
                    for p in paths:
                        sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                        p['rewards'] = sparse_rewards

                train_returns.append(eval_util.get_average_returns(paths))
            train_returns = np.mean(train_returns)
            ### eval train tasks with on-policy data to match eval of test tasks
            train_final_returns, train_online_returns = self._do_eval(indices, epoch, buffer=self.train_buffer)
            eval_util.dprint('train online returns')
            eval_util.dprint(train_online_returns)

            ### test tasks
            eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
            test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch, buffer=self.eval_buffer)
            eval_util.dprint('test online returns')
            eval_util.dprint(test_online_returns)

            # save the final posterior
            self.agent.log_diagnostics(self.eval_statistics)

            if hasattr(self.env, "log_diagnostics"):
                self.env.log_diagnostics(paths, prefix=None)

            avg_train_return = np.mean(train_final_returns)
            avg_test_return = np.mean(test_final_returns)
            avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
            avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
            self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
            self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
            self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return

            self.loss['train_returns'] = train_returns
            self.loss['avg_train_return'] = avg_train_return
            self.loss['avg_test_return'] = avg_test_return
            self.loss['avg_train_online_return'] = np.mean(avg_train_online_return)
            self.loss['avg_test_online_return'] = np.mean(avg_test_online_return)

        logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

    def collect_paths(self, idx, epoch, run, buffer):
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        # num_trajs = 0
        while num_transitions < self.num_steps_per_eval:
            path, num = self.offline_sampler.obtain_samples(
                buffer=buffer,
                deterministic=self.eval_deterministic,
                max_samples=self.num_steps_per_eval - num_transitions,
                max_trajs=1,
                accum_context=True,
                rollout=True)
            paths += path
            num_transitions += num

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, buffer, add_to_enc_buffer=True):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.agent.clear_z()

        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.offline_sampler.obtain_samples(buffer=buffer,
                                                           max_samples=num_samples - num_transitions,
                                                           max_trajs=update_posterior_rate,
                                                           accum_context=False,
                                                           resample=resample_z_rate,
                                                           rollout=False)
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.sample_context(self.task_idx)
                self.agent.infer_posterior(context, task_indices=np.array([self.task_idx]))
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')
    
    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass


