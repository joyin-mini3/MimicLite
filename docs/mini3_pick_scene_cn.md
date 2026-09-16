# Mini3 双夹爪拾取场景

机器人模型：[`mini3_gripper.xml`](../any4hdmi/assets/robots/mini3_mjlab/mini3_gripper.xml)。
完整场景：[`scene_pick_place.xml`](../any4hdmi/assets/robots/mini3_mjlab/scene_pick_place.xml)。
原始 `mini3.xml` 保留，新增模型未替换原训练资产。

## 机器人改动

左右前臂末端各增加 **100 mm 刚性延长段**，之后安装一副平行双指夹爪。
安装起点为原 XML 的肘末端碰撞球心：左侧 `(0.107, -0.018, 0.0085)`，右侧
`(0.107, 0.018, 0.0085)`，均在对应 elbow body 坐标系中沿 **+X** 延长。
100 mm 不包括夹爪长度；夹持 TCP 位于安装座前方 68 mm。

| 项目 | 设定 |
| --- | --- |
| 最大内开口 | 70 mm |
| 每根手指长度 | 70 mm |
| 每侧新增质量 | 0.20 kg：延长段 0.06、掌座 0.08、两指各 0.03 |
| 原关节/驱动 | 原 21 个关节参数和 21 个 motor 名称、驱动顺序保留 |
| 新自由度 | 每侧 2 个 slide joint，以 equality 约束同步 |
| 新控制输入 | 每侧 1 个 position actuator，共 23 个驱动输入 |
| 碰撞与惯性 | 延长段、掌座、手指均有碰撞；新增惯性由形状和质量计算 |

夹爪为简化仿真结构，新增质量、惯性和伺服参数是调试初值，未按某个实体夹爪标定。

控制接口：

```python
# data.ctrl 的下标必须按名字查找，不能假设夹爪关节位于 qpos 的末尾。
data.ctrl[model.actuator("left_gripper_ctrl").id] = 0.035  # 左爪开到 70 mm
data.ctrl[model.actuator("right_gripper_ctrl").id] = 0.0   # 右爪闭合
```

`ctrl` 单位为米，范围 `[0, 0.035]`，内开口为 `2 * ctrl`。
TCP site 分别是 `left_gripper_tcp` 和 `right_gripper_tcp`。
指关节名为 `{side}_gripper_finger_joint` 与 `{side}_gripper_follower_joint`。
机器人模型 `nq=32, nv=31, nu=23`，完整场景 `nq=53, nv=49, nu=23`。

## RGB 摄像头

模型包含头部和左右夹爪三个 RGB 摄像头，固定在对应连杆上并随其运动。

| 名称 | 所属连杆 | 光心在连杆坐标系中的位置（米） | 朝向 |
| --- | --- | --- | --- |
| `head_rgb` | `head_link` | `(0.073, 0, 0.048)` | 相对头部 +X 向前方向下俯 **45°** |
| `left_gripper_rgb` | `left_gripper_palm` | `(0.020, 0, 0.040)` | 指向左爪 TCP / 指间区域，下俯约 39.81° |
| `right_gripper_rgb` | `right_gripper_palm` | `(0.020, 0, 0.040)` | 指向右爪 TCP / 指间区域，下俯约 39.81° |

三个摄像头均设为 **640 × 480 RGB、垂直视场角 80°**。头部相机在头部坐标系中的
光轴为 `(√0.5, 0, -√0.5)`；角度随头部姿态一起变化，不是固定的世界俯仰角。
外壳、镜头和夹爪上的支架仅作显示，质量为 0、不参与碰撞；新增连杆和夹爪的质量仍
为每侧 200 g、双侧共 400 g。相机尚未增加实际硬件的质量或惯性。

```bash
# 直接打开头部 RGB 视角
python preview_mini3_pick_scene.py --camera head_rgb
# 保存一帧 640×480 RGB 图像
python preview_mini3_pick_scene.py --camera head_rgb \
  --screenshot outputs/mini3_pick_scene/head_rgb.png
python preview_mini3_pick_scene.py --camera left_gripper_rgb \
  --screenshot outputs/mini3_pick_scene/left_gripper_rgb.png
python preview_mini3_pick_scene.py --camera right_gripper_rgb \
  --screenshot outputs/mini3_pick_scene/right_gripper_rgb.png
```

自己的仿真循环中可直接获取 RGB 数组（`uint8`，形状 `[480, 640, 3]`）：

```python
option = mujoco.MjvOption()
option.geomgroup[1] = 0  # 隐藏原机器人的碰撞代理显示
with mujoco.Renderer(model, height=480, width=640) as renderer:
    renderer.update_scene(data, camera="head_rgb", scene_option=option)
    rgb = renderer.render()
```

## 物件布局

默认初始化骨盆位于 `(0, 0, 0.46305)`，机器人面向世界 **+X**。
物件在初始化时根据骨盆的平面位置和 yaw 放置，之后保留在世界坐标中，不随机器人移动。

| 物件 | 相对初始化骨盆的平面坐标（米） | 尺寸/物理属性 |
| --- | --- | --- |
| 红色方块 `pick_cube_1` | `(0.28, 0.12)` | 边长 4 cm，质量 40 g，自由刚体 |
| 绿色方块 `pick_cube_2` | `(0.33, 0)` | 同上 |
| 蓝色方块 `pick_cube_3` | `(0.28, -0.12)` | 同上 |
| 篮子 `pick_basket` | `(0.52, 0)` | 内部 24 × 22 cm，外高 12 cm，底板/墙厚 1 cm |

方块初始中心高 2.05 cm，释放后落在地面上。篮子固定在地面，有真实底板和四面碰撞墙，
上方开放。篮子目标 site 为 `pick_basket_target`。

## 打开和手动调试

使用已配置的 MJLab Python 环境，在仓库根目录运行：

```bash
source active-adaptation/venv/mjlab/.venv/bin/activate
python preview_mini3_pick_scene.py
```

| 按键 | 功能 |
| --- | --- |
| `[` / `]` | 选择前一个/后一个原机器人关节，名称打印在终端 |
| 上 / 下方向键 | 所选关节目标增加/减少 0.05 rad |
| `Z` / `X` | 左夹爪打开/闭合 |
| `C` / `V` | 右夹爪打开/闭合 |
| `B` | 开关骨盆支撑 |
| `W` / `S`，`A` / `D` | 支撑目标沿世界 X、Y 轴移动，每次 1 cm |
| `U` / `J` | 支撑目标升/降，每次 1 cm |
| `I` / `K` | 支撑目标绕自身 Y 轴俯仰，每次 3° |
| `F8` / `F9` / `F10` / `F11` | 总览 / 头部 RGB / 左爪 RGB / 右爪 RGB |
| `P` | 暂停/继续物理仿真 |
| `R` | 重置机器人、方块、关节目标及电机状态 |
| `Esc` | 关闭窗口 |

预览默认启用 **骨盆支撑夹具**，方便手动调节；`--no-support` 可关闭。
场景 XML 中 `base_support_weld` 默认是 **inactive**，外部程序直接加载时机器人自由浮动。
原 21 关节使用 Mini3 Real Motor 模型和关节位置 PD，夹爪使用新加的理想位置伺服器。
窗口默认隐藏原碰撞代理的可视几何，碰撞物理仍生效。

机器人直立时仍够不到地面，需要协调腿部、躯干和上肢姿态。仅降低骨盆而不屈腿会把脚
压向地面。此入口用于场景和关节调试，不提供自主平衡或自动拾取策略。
原 MimicLite checkpoint 的 21 关节部署入口不能直接加载该扩展模型；后续接入 policy
时需要按名字分离 21 个原动作与夹爪控制，并评估延长段/新增质量的影响。

```bash
# 无窗口物理检查
python preview_mini3_pick_scene.py --headless --duration 5
# 初态截图，需要可用的 OpenGL 上下文
python preview_mini3_pick_scene.py --screenshot outputs/mini3_pick_scene/scene.png
```

## 重新生成和修改初始朝向

```bash
python any4hdmi/scripts/generate_mini3_gripper.py
python any4hdmi/scripts/generate_mini3_pick_scene.py
# 可输出另一个场景；方块、篮子会跟随初始位置和朝向放置
python any4hdmi/scripts/generate_mini3_pick_scene.py \
  --x 1 --y 2 --yaw-deg 90 --output outputs/mini3_pick_scene/rotated_scene.xml
```

场景 XML 是生成时对机器人 XML 的完整复制再添加道具；更新机器人模型后，需要重新
运行场景生成器。生成文件通过相对路径引用原 meshes，复制到其他机器时也需携带这些 mesh。

验证命令：

```bash
python -m unittest discover -s any4hdmi/tests -p 'test_mini3_gripper.py' -v
python -m unittest discover -s any4hdmi/tests -p 'test_mini3_pick_scene.py' -v
python -m unittest discover -s any4hdmi/tests -p 'test_mini3_rgb_cameras.py' -v
```

物理测试包括左右夹爪分别夹持自由方块并在重力下保持、张开释放、三个方块落地稳定、
篮底承接自由落体、四面墙阻挡，以及初始位置/yaw 改变后的布局一致性。
摄像头测试核对安装朝向、随连杆运动、镜头前方光路以及质量/惯性不变。
