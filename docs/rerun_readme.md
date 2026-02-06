# Rerun using example guide

This guide provides description of how to use rerun to visualize data in this repository. Before using this document you need to install project with [installation guide](installation_guide.md). 

**What is Rerun?**

Rerun is the interactive logger that can be used for visualization of different type of data. The is 2 steps of using it: 

1. [**Data preporation**](#Data-preporation) – you need to create .rrd file that store all the data that is needed to see
2. [**Data visualization**](#Data-visualization) – opening .rrd file in visualisation api

## Data preporation

The main idea of data preporation is to use rerun-sdk library to log the data and create .rrd file using python script.

### The main example

The main example is how to create .rrd file is located in the *examples/rerun_minimal_demo.py* file. The are several steps to log data:

**Library import**

```python
import rerun as rr
```

**File creation and initialisation**

```python
# Init Rerun and save to file
rr.init(args.app_id)
rr.save(str(args.output))
```

**Time setting**

```python
rr.set_time("time", timestamp=0)
```

Rerun file is the sequence of scenes that differ by time tag. If you want to start logging a new scene — change time.

**Transform setting**

```python
rr.log("world/lidar", rr.Transform3D(translation=pose[:3], rotation=rr.Quaternion(xyzw=pose[3:])))
```

Each part of scene data is logged in *world* volume and can be located in adress like *world/adress*. Each adress has its Transform3D state that describe the transformation between global and local coordinate systems. Transformation can be set in different formats (in example we use translation-quaternion format).

**Data logging**

```python
rr.log("world/lidar", rr.Points3D(coords, colors=np.array([200, 200, 0]).astype(np.uint8), radii=0.02))
```

That command adds points from the *coords* 3d numpy array to scene and sets points color and size.

**End .rrd file creation**

```python
rr.disconnect()
```

### How to use the script

To use example you need to know the pass to map with lidar and camera data and use it when running the script:

```bash
uv run python examples/rerun_minimal_demo.py --map-dir=my_map_dir_path --output==convenient_path/result.rrd
```

Also you can choose the path to transformation matrix that transform your results to other coordinate system. Other sensors can be added (from list: lidar, cam_pinhole_left, cam_pinhole_right, cam_fish-eye_left, cam_fish-eye-right) to see more views of one place

**As the result** you receive an .rrd file with all collected data for visualization

## Data visualization

You need to install rerun-sdk inside your system and run it with command:

```bash
rerun
```

It runs the visualization api where you can open your .rrd file manually.

This part of rerun using process can be proceeded seperatly from previous part (on another device, for example). Attention! .rrd file can have very big size so if you want to see data on other device, check its size and chose data transportation way that provide adequate time delay.

**Congratulation!** Now you know how to use rerun to visualize your data and experiments!