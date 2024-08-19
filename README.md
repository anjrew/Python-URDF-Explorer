# Python URDF Explorer

When working with xacro files for some URDF tools they only accept URDF in the pure form so we must first convert them with the xacro tool.

Convert xacro to urdf:

```bash
xacro xacro.py my_robot.xacro > my_robot.urdf
```

Be careful with the paths in the xacro file, they must be relative to the xacro file itself.