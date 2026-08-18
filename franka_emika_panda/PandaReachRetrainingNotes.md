# Panda Reach-and-Hold Retraining Notes (ppo_final_lower_vib)

Actionable design guidance for whoever owns the training repo. The training
environment/config does **not** live on this machine, so this doc describes
*what* to change and *why*, with concrete reward/observation/config sketches.
For the measured deployment behavior, pipeline, and terminology this references,
see `franka_emika_panda/PandaReachPolicyDeploymentNotes.md`.

## 1. Problem Summary — Why This Must Be Fixed In Training

The deployed `ppo_final_lower_vib` policy reaches the target fine but is
effectively **bang-bang at the hold point**:

- Near the goal the raw policy action jumps **~7–9 rad (L2) every control tick**
  and commands a joint target **~4 rad away from the current pose each tick**.
- On the robot, the action rate/velocity limiter (see deployment notes:
  smoothing `alpha=0.20`, 60 Hz rate limit, velocity limits
  `~[0.744]*4,[0.893]*3 rad/s`, measured-velocity governor at `0.75`) clips
  **100% of joint-ticks near the goal**. The applied command is pinned at the
  slew cap (**~0.027 rad/tick ≈ 1.6 rad/s**) and reverses direction each tick →
  a **~10 Hz saturated limit cycle** (visible wrist / joint-7 oscillation
  ~8 deg peak-to-peak). TCP position is steady (~6 mm).
- Deployment-side linear filtering (per-joint notch, smoothing, rate limiting)
  removes only **~11%** of the vibration because it acts on an *already
  saturated* limit-cycle signal.

Root cause is a **sim-to-deploy gap in the objective**: training rewards
reaching and holding TCP position, but does **not** penalize the
high-frequency raw-action churn. Because the deployment limiter saturates, the
policy pays no cost in sim for emitting huge oscillating raw actions — TCP still
looks steady in sim. The fix must change the *training objective and the signal
the policy is optimized against*. Filtering downstream of a saturated actuator
cannot recover it.

## 2. Match The Deployment Action Pipeline In Training (Most Important Change)

Optimize the policy against the **actually-applied command**, not the raw
action. Apply the **same** action processing during training that runs on the
robot, in the same order, every step:

1. interpret raw action as absolute joint position target (radians),
2. first-order smoothing: `filtered = lerp(prev_filtered, raw, alpha)` with the
   **deployed `alpha=0.20`**,
3. per-joint rate limit at the deployed rate (`60 Hz`) → per-tick slew cap
   (`~0.027 rad/tick`),
4. per-joint velocity clamp (`~[0.744]*4,[0.893]*3 rad/s`),
5. measured-velocity governor (`start_ratio=0.75`),
6. joint-limit clip.

Then step MuJoCo with the **governed** command, and compute reward from that
governed command / resulting joint state — not the raw action.

```python
# pseudo-code inside env.step(raw_action)
applied = self.action_pipeline(raw_action, self.prev_applied, self.qvel)  # steps 1-6
self.sim.set_ctrl(applied)
self.sim.step()
obs, rew = self.observe(), self.reward(raw_action, applied, ...)
self.prev_applied = applied
```

This is the single most important change. With the limiter *inside* the
training loop, the policy can no longer "win" by emitting saturating actions:
the sim will show the same 10 Hz limit cycle, so any smoothness reward term
(Section 3) now has gradient to act on. Make sure the training control rate and
`dt` match deployment so per-tick slew caps line up.

## 3. Reward Shaping Additions

Add smoothness/effort terms on top of the existing reach + hold + settled-bonus
reward. Compute penalties on the **raw action** (that is the churn to kill), and
optionally also on the applied command. Suggested terms and starting weight
ranges (tune per your reward scale):

- **Action rate penalty** — `-w1 * ||a_t - a_{t-1}||^2`, `w1 ≈ 0.01–0.1`.
  Directly targets the ~7–9 rad per-tick jump.
- **Action jerk / second-difference penalty** —
  `-w2 * ||a_t - 2 a_{t-1} + a_{t-2}||^2`, `w2 ≈ 0.005–0.05`. Kills the
  direction-reversal that drives the limit cycle.
- **Near-goal stillness** — penalize joint velocity (or command delta) weighted
  more heavily when TCP error is small (the hold phase):
  `-w3 * gate * ||qvel||^2`, with `gate = 1` when `tcp_dist < 0.05 m` (or a
  smooth `exp(-tcp_dist/d0)`), `w3 ≈ 0.02–0.2`. This is where the vibration
  lives, so weight it harder here than during approach.
- **Effort / action-magnitude penalty** — `-w4 * ||a_t||^2` (or relative to
  current joint pose), `w4 ≈ 0.001–0.01`. Discourages large absolute targets.

```python
rate  = -w1 * np.sum((a_t - a_tm1)**2)
jerk  = -w2 * np.sum((a_t - 2*a_tm1 + a_tm2)**2)
gate  = 1.0 if tcp_dist < 0.05 else np.exp(-tcp_dist / 0.05)
still = -w3 * gate * np.sum(qvel**2)
effort= -w4 * np.sum(a_t**2)
reward = reach_and_hold + rate + jerk + still + effort
```

Tuning guidance: **start small and scale up.** Begin with `w1`/`w2` near the low
end and confirm success/settled-success rate stays at `1.0`, then increase in
2× steps while watching (a) reach time / overshoot, (b) success and settled
success, (c) the raw-action per-tick delta L2 near goal (Section 6). Stop
raising a weight as soon as reach performance starts to degrade; back off ~30%.
Add the near-goal stillness term *after* rate/jerk are stable, since it is the
most likely to cause sluggish approach if the gate leaks into the approach
phase.

## 4. Observation Changes

The current 38-dim observation **already includes the previous applied
command** (`prev_action` = previous governed target `/ pi`) and per-joint
`qpos, qvel, prev_qpos, prev_qvel`. That is good and should be kept — it is what
lets the policy learn smooth incremental deltas instead of jumping to absolute
targets.

Recommended:

- **Keep `prev_action` as the previous *applied/governed* command**, not the raw
  policy output (matches deployment; do not switch it to raw). This is what
  makes incremental/delta behavior learnable.
- If you move to delta actions (Section 5), the previous applied command in the
  obs is essential — verify it stays wired to the governed command after the
  pipeline change in Section 2.
- Optionally add the **previous *two* raw actions** (or `a_{t-1}-a_{t-2}`) to the
  obs so the jerk term is fully observable/Markovian. Only needed if the jerk
  penalty struggles to converge.

Why it matters: without previous-action context the policy must re-derive an
absolute target each tick from state alone, which encourages large corrective
jumps. With it, small residual deltas are representable and cheap.

## 5. Action Space / Policy Considerations

- **Prefer delta / incremental actions** relative to the current joint position:
  `target = qpos + scale * a_t`, with a small `scale` (e.g. sized so full-scale
  action ≈ the deployment slew cap of `~0.027 rad/tick`, or a small multiple).
  This structurally bounds per-tick command change and makes "hold" the
  zero-action fixed point instead of a high-frequency balancing act.
- If keeping absolute targets, **reduce action scale** so the reachable per-tick
  target change is closer to the slew cap; a huge action range invites
  saturation.
- **Reduce exploration std late in training** (entropy coefficient decay or std
  annealing). High policy std at convergence directly injects the tick-to-tick
  action noise that becomes the limit cycle.
- **Evaluate and export deterministically** (mean action, no sampling) — the
  ONNX export path already exports the deterministic policy; confirm this.
- **Optional curriculum "hold" phase:** after the arm settles, lower control
  authority near the goal (shrink action scale or tighten the velocity clamp
  once `tcp_dist < 0.05 m` for `SettledDwellRequiredSteps = 20`), so the policy
  learns a low-authority hold rather than fighting the limiter.

## 6. Validation Checklist

In the training env (with the deployment pipeline from Section 2 active):

- [ ] Log **raw-action per-tick delta L2** (`||a_t - a_{t-1}||`) near the goal;
      confirm it drops **well below the deployment slew cap** (~0.027 rad/tick),
      i.e. no longer 7–9 rad. This is the primary sim-side success signal.
- [ ] Log the **rate-limiter saturation fraction** near goal; target **well
      below 100%** (ideally near 0% at hold).
- [ ] Confirm **success rate and settled success rate** stay ~`1.0` and hold is
      stable (TCP steady, `qvel` L2 under the `0.4 rad/s` success cap).
- [ ] Check reach time / overshoot did not regress vs the current baseline.

After export:

- [ ] Export deterministically to ONNX and re-run on the deployment telemetry
      (see deployment notes) with **no** extra ROS filtering beyond safety
      clamps.
- [ ] Recheck near-goal deployment metrics vs the baseline: **`qvel` L2 rms/p95**
      (baseline no-notch `1.08 / 1.32`), **joint-7 peak-to-peak** (baseline
      ~8 deg), and **rate-limiter saturation fraction** (baseline ~100%). Target
      a large drop in all three, with reach success preserved.
- [ ] Only then compare against the linear-filter deployment candidates; a
      retrained policy should beat the ~11% filtering ceiling by a wide margin.
