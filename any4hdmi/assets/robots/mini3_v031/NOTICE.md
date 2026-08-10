# Mini3 v0.3.1 asset

This directory is a verbatim, versioned copy of the locally supplied
`mini3_v0.3.1` robot package. It is intentionally separate from
`mini3_mjlab`: the two variants share a 21-DoF joint contract but differ in
joint limits, default base height, and collision geometry.

Use `mini3-v031` for newly trained policies. Existing `mini3` checkpoints and
datasets remain bound to the legacy asset and must not be mixed with this one.
