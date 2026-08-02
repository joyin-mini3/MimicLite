# Mini3 MuJoCo asset

This directory was copied from:

`UFO/humanoidverse/data/robots/mini3_mjlab`

The Isaac Lab conversion source at `urdf/mini3.urdf` was copied from:

`UFO/humanoidverse/data/robots/mini3/urdf/mini3.urdf`

The bundled UFO license is preserved in
`LICENSE-UFO-CC-BY-NC-4.0.txt`. It is Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0); do not assume
commercial-use permission.

Local modification: collision geoms in `mini3.xml` use `contype="1"` so
self-collision is enabled. The asset's explicit adjacent-link `<exclude>`
entries remain in effect.
