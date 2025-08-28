metric["update_steps"] = update_steps
update_steps = update_steps + 1 

# 新增：在 JIT 里计算并塞到 metric，避免回调里再动 Tracer
metric["debug_adv_std"] = advantages.std()
metric["debug_valid_action_sum"] = traj_batch.valid_action.sum()
metric["debug_valid_action_ratio"] = traj_batch.valid_action.sum() / traj_batch.valid_action.size












writer.add_scalar('eval/episodic_length', ep_len, env_steps)
writer.add_scalar('eval/success_rate', succ_rate, env_steps)
writer.add_scalar('eval/alive_count', alive_cnt, env_steps)
# 原：advantages.std() / traj_batch.valid_action.sum()（会泄漏 Tracer）
# 改：使用 metric 中的宿主标量
writer.add_scalar('debug/adv_std', metric["debug_adv_std"], env_steps)
writer.add_scalar('debug/valid_action_sum', metric["debug_valid_action_sum"], env_steps)
writer.add_scalar('debug/valid_action_ratio', metric["debug_valid_action_ratio"], env_steps)