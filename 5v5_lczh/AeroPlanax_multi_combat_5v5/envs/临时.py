# def tactical_position_reward_fn(
#     state: HierarchicalCombatTaskState,
#     params: HierarchicalCombatTaskParams,
#     agent_id: AgentID,
#     reward_scale: float = 1.0,
#     num_allies: int = 1,
#     num_enemies: int = 2,
# ) -> float:
#     """
#     计算战术位置奖励：
#     - rear_hemisphere_reward：TA 越小越好（在敌机后半球）
#     - distance_reward：与“最佳距离”的高斯带状越近越好
#     - altitude_reward：绝对高度在[min,max]的中带附近越好（与敌机无关）
#     """
#     def safe_exp(x):
#         return jnp.exp(jnp.clip(x, -10.0, 10.0))

#     # ego 的6维位置信息+速度
#     ego_feature = jnp.hstack((
#         state.plane_state.north[agent_id],
#         state.plane_state.east[agent_id],
#         state.plane_state.altitude[agent_id],
#         state.plane_state.vel_x[agent_id],
#         state.plane_state.vel_y[agent_id],
#         state.plane_state.vel_z[agent_id]
#     ))

#     # 目标列表：若是我方，取所有敌机；若为敌方，取所有我方
#     enm_list = jax.lax.select(
#         agent_id < num_allies,
#         jnp.arange(num_allies, num_allies + num_enemies),
#         jnp.arange(num_allies)
#     )

#     def compute_tactical_reward(enm_id):
#         # 敌机6维位置信息+速度
#         enm_feature = jnp.hstack((
#             state.plane_state.north[enm_id],
#             state.plane_state.east[enm_id],
#             state.plane_state.altitude[enm_id],
#             state.plane_state.vel_x[enm_id],
#             state.plane_state.vel_y[enm_id],
#             state.plane_state.vel_z[enm_id]
#         ))

#         # 相对几何量（AO/TA/R）
#         AO, TA, R, _ = get_AO_TA_R(ego_feature, enm_feature)
#         AO = jnp.clip(AO, 0.0, jnp.pi)
#         TA = jnp.clip(TA, 0.0, jnp.pi)

#         # 1) 后半球：TA 越小越好（指数衰减）
#         rear_hemisphere_reward = safe_exp(-TA / (jnp.pi / 4))

#         # 2) 距离：围绕最佳距离（max_distance/3）高斯带状
#         optimal_distance = params.max_distance / 3
#         distance_denominator = jnp.maximum(2 * (optimal_distance/2)**2, 1e-8)
#         distance_reward = safe_exp(-(R - optimal_distance)**2 / distance_denominator)

#         # 3) 绝对高度带状（与敌机无关）：靠近中带 (min,max 的中点) 越好
#         mid = (params.max_altitude + params.min_altitude) / 2.0
#         band = (params.max_altitude - params.min_altitude) / 4.0
#         altitude_reward = jnp.exp(-((state.plane_state.altitude[agent_id] - mid) / (band + 1e-6))**2)

#         # 加权求和（固定权重 0.5/0.45/0.05）
#         tactical_reward = (rear_hemisphere_reward * 0.5 +
#                            distance_reward * 0.45 +
#                            altitude_reward * 0.05)

#         # 只对“活着或被锁定”的敌机计分
#         alive_or_locked = state.plane_state.is_alive[enm_id] | state.plane_state.is_locked[enm_id]
#         tactical_reward = jnp.where(jnp.isnan(tactical_reward) | jnp.isinf(tactical_reward), 0.0, tactical_reward)
#         return tactical_reward * alive_or_locked

#     # 对所有目标“求和”
#     per_target_rewards = jax.vmap(compute_tactical_reward)(enm_list)
#     total_reward = jnp.sum(per_target_rewards)

#     # 自己也要活着/被锁定才记分
#     mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
#     final_reward = total_reward * reward_scale * mask
#     final_reward = jnp.where(jnp.isnan(final_reward) | jnp.isinf(final_reward), 0.0, final_reward)
#     return final_reward









                        grad_fn = jax.value_and_grad(main_loss_fn, has_aux=True)
                        main_loss, grads = grad_fn(
                            network_train_state.params, network_hstate, traj_batch, advantages, targets
                        )
                        # 计算actor梯度统计
                        actor_grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
                        # 获取最大梯度
                        actor_grad_abs_max = jnp.max(jnp.array([jnp.max(jnp.abs(g)) for g in jax.tree_util.tree_leaves(grads)]))

+                       # 监控坏梯度（actor）
+                       actor_bad_grad = sum([(~jnp.isfinite(g)).sum() for g in jax.tree_util.tree_leaves(grads)]).astype(jnp.float32)
+                       actor_nan_grad = sum([jnp.isnan(g).sum() for g in jax.tree_util.tree_leaves(grads)]).astype(jnp.float32)
+                       actor_inf_grad = sum([jnp.isinf(g).sum() for g in jax.tree_util.tree_leaves(grads)]).astype(jnp.float32)

                        network_train_state = network_train_state.apply_gradients(grads=grads)

                        if ENABLE_CRITIC:
                            critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
                            critic_loss, critic_grads = critic_grad_fn(
                                critic_network_train_state.params, critic_network_hstate, traj_batch, targets
                            )
                            # 计算critic梯度统计
                            critic_grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(critic_grads)))
                            # 获取最大梯度
                            critic_grad_abs_max = jnp.max(jnp.array([jnp.max(jnp.abs(g)) for g in jax.tree_util.tree_leaves(critic_grads)]))
+                           # 监控坏梯度（critic）
+                           critic_bad_grad = sum([(~jnp.isfinite(g)).sum() for g in jax.tree_util.tree_leaves(critic_grads)]).astype(jnp.float32)
+                           critic_nan_grad = sum([jnp.isnan(g).sum() for g in jax.tree_util.tree_leaves(critic_grads)]).astype(jnp.float32)
+                           critic_inf_grad = sum([jnp.isinf(g).sum() for g in jax.tree_util.tree_leaves(critic_grads)]).astype(jnp.float32)

                            critic_network_train_state = critic_network_train_state.apply_gradients(grads=critic_grads)
                            
                            total_loss = main_loss[0] + critic_loss[0]
                            loss_info = {
                                "total_loss": total_loss,
                                "actor_loss": main_loss[0],
                                "value_loss": critic_loss[0],
                                "entropy": main_loss[1][1],
                                "ratio": main_loss[1][2],
                                "approx_kl": main_loss[1][3],
                                "clip_frac": main_loss[1][4],
                                "actor_grad_norm": actor_grad_norm,
                                "actor_grad_max": actor_grad_abs_max,
                                "critic_grad_norm": critic_grad_norm,
                                "critic_grad_max": critic_grad_abs_max,
+                               # 坏梯度监控汇总与分解
+                               "bad_grad": actor_bad_grad + critic_bad_grad,
+                               "bad_grad_actor": actor_bad_grad,
+                               "bad_grad_critic": critic_bad_grad,
+                               "nan_grad_actor": actor_nan_grad,
+                               "inf_grad_actor": actor_inf_grad,
+                               "nan_grad_critic": critic_nan_grad,
+                               "inf_grad_critic": critic_inf_grad,
                            }
                        else:
                            loss_info = {
                                "total_loss": main_loss[0],
                                "value_loss": main_loss[1][0],
                                "actor_loss": main_loss[1][1],
                                "entropy": main_loss[1][2],
                                "ratio": main_loss[1][3],
                                "approx_kl": main_loss[1][4],
                                "clip_frac": main_loss[1][5],
                                "actor_grad_norm": actor_grad_norm,
                                "actor_grad_max": actor_grad_abs_max,
+                               # 仅 actor 的坏梯度
+                               "bad_grad": actor_bad_grad,
+                               "bad_grad_actor": actor_bad_grad,
+                               "nan_grad_actor": actor_nan_grad,
+                               "inf_grad_actor": actor_inf_grad,
                            }





-                        # 全局数值清理，防止 NaN / Inf 流入 logger
-                        loss_info = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), loss_info)
-                        bad_grad = sum([(~jnp.isfinite(g)).sum() for g in jax.tree_util.tree_leaves(grads)])
-                        
-                        return (network_train_state, critic_network_train_state), loss_info
+                        # 全局数值清理，防止 NaN / Inf 流入 logger
+                        loss_info = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), loss_info)
+                        return (network_train_state, critic_network_train_state), loss_info










                        # 将梯度统计单独写到一个新的分组
                        grad_metrics = {
                            'actor_grad_norm': metric["loss"]["actor_grad_norm"],
                            'actor_grad_max': metric["loss"]["actor_grad_max"]
                        }
                        
                        if "critic_grad_norm" in metric["loss"]:
                            grad_metrics.update({
                                'critic_grad_norm': metric["loss"]["critic_grad_norm"],
                                'critic_grad_max': metric["loss"]["critic_grad_max"]
                            })
+
+                        # 坏梯度监控（存在即写）
+                        if "bad_grad" in metric["loss"]:
+                            grad_metrics['bad_grad'] = metric["loss"]["bad_grad"]
+                        if "bad_grad_actor" in metric["loss"]:
+                            grad_metrics['bad_grad_actor'] = metric["loss"]["bad_grad_actor"]
+                        if "bad_grad_critic" in metric["loss"]:
+                            grad_metrics['bad_grad_critic'] = metric["loss"]["bad_grad_critic"]
+                        if "nan_grad_actor" in metric["loss"]:
+                            grad_metrics['nan_grad_actor'] = metric["loss"]["nan_grad_actor"]
+                        if "inf_grad_actor" in metric["loss"]:
+                            grad_metrics['inf_grad_actor'] = metric["loss"]["inf_grad_actor"]
+                        if "nan_grad_critic" in metric["loss"]:
+                            grad_metrics['nan_grad_critic'] = metric["loss"]["nan_grad_critic"]
+                        if "inf_grad_critic" in metric["loss"]:
+                            grad_metrics['inf_grad_critic'] = metric["loss"]["inf_grad_critic"]
                             
                        for k, v in grad_metrics.items():
                            v = jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                            writer.add_scalar('grad/{}'.format(k), v, env_steps)