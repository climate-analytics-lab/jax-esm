


def trajectory_from_step(
    step_fn: TimeStepFn,
    outer_steps: int,
    inner_steps: int,
    *,
    start_with_input: bool = False,
    post_process_fn: PostProcessFn = lambda x: x,
    outer_scan_fn: typing.ScanFn = jax.lax.scan,
    inner_scan_fn: typing.ScanFn = jax.lax.scan,
) -> Callable[[PyTreeState], tuple[PyTreeState, Any]]:
  """Returns a function that accumulates repeated applications of `step_fn`.

  Compute a trajectory by repeatedly calling `step_fn()`
  `outer_steps * inner_steps` times.

  Args:
    step_fn: function that takes a state and returns state after one time step.
    outer_steps: number of steps to save in the generated trajectory.
    inner_steps: number of repeated calls to step_fn() between saved steps.
    start_with_input: if True, output the trajectory at steps [0, ..., steps-1]
      instead of steps [1, ..., steps].
    post_process_fn: function to apply to trajectory outputs.
    outer_scan_fn: scan function to use for outer (saved) steps.
    inner_scan_fn: scan function to use for inner (unsaved) steps.

  Returns:
    A function that takes an initial state and returns a tuple consisting of:
      (1) the final frame of the trajectory.
      (2) trajectory of length `outer_steps` representing time evolution.
  """
  if inner_steps != 1:
    step_fn = repeated(step_fn, inner_steps, inner_scan_fn)

  def step(carry_in, _):
    carry_out = step_fn(carry_in)
    frame = carry_in if start_with_input else carry_out
    return carry_out, post_process_fn(frame)

  def multistep(x):
    return outer_scan_fn(step, x, xs=None, length=outer_steps)

  return multistep