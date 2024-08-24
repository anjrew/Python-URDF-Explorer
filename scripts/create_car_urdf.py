
def create_boticar_urdf() -> URDF:
    chassis_link = create_chassis_link(robot_properties.chassis)

    front_wheel_x_m_distance_from_chassis_center = robot_properties.chassis.wheelbase / 2
    wheel_z_value_from_chassis_center = -robot_properties.chassis.wheel_z_base_offset_m

    assemblies = {}
    for position in WheelPosition:
        is_left = 'LEFT' in position.name
        is_front = 'FRONT' in position.name
        name_prefix = position.name.lower()

        wheel = create_wheel_link(name_prefix, robot_properties.wheel)
        axle = create_axle_link(name_prefix, robot_properties.axle)
        steering_column = create_wheel_shock_mount_column_link(name_prefix, robot_properties.chassis.wheel_mounts)

        wheel_axle_joint = create_wheel_axle_joint(wheel.name, axle.name)
        axle_column_joint = create_axle_column_joint(axle.name, steering_column.name, robot_properties.axle)

        y_offset = robot_properties.chassis.steering_column_y_center_offset_m * (1 if is_left else -1)
        x_offset = front_wheel_x_m_distance_from_chassis_center * (1 if is_front else -1)
        rotation = math.pi/2 if is_left else 3*math.pi/2

        mount = (x_offset, y_offset, wheel_z_value_from_chassis_center, 0, 0, rotation)
        
        if is_front:
            column_chassis_joint = create_steering_column_chassis_joint(
                CHASSIS_ID, steering_column.name, robot_properties.steering_column, mount
            )
        else:
            column_chassis_joint = create_fixed_column_chassis_joint(
                CHASSIS_ID, steering_column.name, mount
            )

        assemblies[position] = WheelAssembly(
            wheel, axle, steering_column,
            wheel_axle_joint, axle_column_joint, column_chassis_joint
        )

    boticar = Boticar(chassis_link, assemblies)

    return URDF(name="Boticar", links=boticar.links, joints=boticar.joints)

# Create and visualize the URDF
boticar_urdf = create_boticar_urdf()
boticar_urdf.show()

# Optionally, animate the URDF
boticar_urdf.animate(cfg_trajectory={
    joint.name: [-math.pi / 4, math.pi / 4] 
    for joint in boticar_urdf.joints 
    if joint.joint_type in [JointType.REVOLUTE.value, JointType.CONTINUOUS.value]
})