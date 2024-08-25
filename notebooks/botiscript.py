# %%
import os
import sys

# Add the root directory for packages
notebook_dir = os.getcwd()

project_root_dir = notebook_dir


sys.path.append(project_root_dir)


# %%

from urdfpy import URDF, Box, Cylinder, Joint, Link, Material, Collision, Visual, Inertial, Geometry


# %%
from typing import Tuple

import numpy as np

from urdfpy.urdf import JointLimit
from urdfpy.utils import xyz_rpy_to_matrix

# %%
# Get properties via a yaml file
from typing import Any, Dict
import yaml

def load_yaml_properties(file_path) -> Dict[str, Any]:
    with open(file_path, 'r') as file:
        properties = yaml.safe_load(file)
        return properties

# %%
MODEL_SCALER=10

# %%
config_path = os.path.join(project_root_dir, 'config', 'config.yaml')

# %%
properties = load_yaml_properties(config_path)

# %%


from pydantic import BaseModel, Field
from typing import Dict

class Angles(BaseModel):
    degrees_45: float = Field(..., description="45 degrees in radians")
    degrees_90: float = Field(..., description="90 degrees in radians")

    
class WheelMount(BaseModel):
    length_m: float = Field(..., description="Vertical height of the mount in meters")
    diameter_m: float = Field(..., description="Diameter of the mount in meters")
    mass_kg: float = Field(..., description="Mass of the mount in kilograms")
    
class Chassis(BaseModel):
    length_m: float = Field(..., description="Chassis length in meters")
    width_m: float = Field(..., description="Chassis width in meters")
    height_m: float = Field(..., description="Chassis height in meters")
    mass_kg: float = Field(..., description="Chassis mass in kilograms")
    wheelbase: float = Field(..., description="Wheelbase in meters")
    wheel_x_offset_from_edge_m: float = Field(..., description="Wheel X offset from front and back edge in meters")
    wheel_z_base_offset_m: float = Field(..., description="Wheel Z base offset in meters")
    steering_column_y_center_offset_m: float = Field(..., description="Steering column Y offset from the side of the chassis in meters")
    wheel_mounts: WheelMount = Field(..., description="Details for the wheel mounts properties")

class Wheel(BaseModel):
    dimeter_m: float = Field(..., description="Wheel diameter in meters")
    width_m: float = Field(..., description="Wheel width in meters")
    mass_kg: float = Field(..., description="Wheel mass in kilograms")

class Tire(BaseModel):
    diameter_m: float = Field(..., description="Tire diameter in meters")
    depth_m: float = Field(..., description="Tire depth in meters")
    width_m: float = Field(..., description="Tire width in meters")


class SteeringColumn(BaseModel):
    max_angle_deg: float = Field(..., description="Maximum steering angle in degrees")
    effort_Nm: float = Field(..., description="Maximum torque that can be applied to the steering column in Newton-meters")
    rads_per_sec: float = Field(..., description="Maximum angular velocity of the steering column in radians per second")
    front_camber_deg: float = Field(..., description="Camber angle in degrees")

class CarGeometry(BaseModel):
    rotational_center_offset_m: float = Field(..., description="Rotational center offset in meters")

class LidarBase(BaseModel):
    height_m: float = Field(..., description="Lidar base height in meters")
    width_m: float = Field(..., description="Lidar base width in meters")

class LidarScanner(BaseModel):
    diameter_m: float = Field(..., description="Lidar scanner diameter in meters")
    height_m: float = Field(..., description="Lidar scanner height in meters")

class Lidar(BaseModel):
    base: LidarBase
    scanner: LidarScanner
    weight_kg: float = Field(..., description="Lidar weight in kilograms")

class LidarMount(BaseModel):
    height_m: float = Field(..., description="Lidar mount height in meters")
    width_m: float = Field(..., description="Lidar mount width in meters")
    length_offset_m: float = Field(..., description="Lidar mount length offset in meters")
    mass_kg: float = Field(..., description="Lidar mount mass in kilograms")

class Mounts(BaseModel):
    lidar: LidarMount

class AxleProperties(BaseModel):
    length_m: float = Field(..., description="Length of the axle in meters")
    diameter_m: float = Field(..., description="Diameter of the axle in meters")
    mass_kg: float = Field(..., description="Mass of the axle in kilograms")
    wheel_inset_m: float = Field(..., description="For far the wheel is inset from the edge of the axle in meters")
    
class Boticar(BaseModel):
    chassis: Chassis
    wheel: Wheel
    axle: AxleProperties
    steering_column: SteeringColumn
    tire: Tire
    geometry: CarGeometry
    lidar: Lidar
    mounts: Mounts


# %%
robot_properties = Boticar.model_validate(properties["boticar"])

# %%
class ColorsRgba:
    BLACK = (0, 0, 0, 1)
    WHITE = (1, 1, 1, 1)
    RED = (1, 0, 0, 1)
    GREEN = (0, 1, 0, 1)
    BLUE = (0, 0, 1, 1)
    YELLOW = (1, 1, 0, 1)
    CYAN = (0, 1, 1, 1)
    MAGENTA = (1, 0, 1, 1)
    GREY = (0.5, 0.5, 0.5, 1)  # Medium grey
    LIGHT_GREY = (0.75, 0.75, 0.75, 1)
    DARK_GREY = (0.25, 0.25, 0.25, 1)
    PINK = (1, 0.75, 0.8, 1)

# %%

from enum import Enum


class JointType(str, Enum):
    """
    Enum representing the types of joints used in robotics.
    """

    FIXED = "fixed"
    """
    A fixed joint has no degrees of freedom. It rigidly attaches two links together.
    Use this for parts that do not move relative to each other.
    """

    REVOLUTE = "revolute"
    """
    A revolute joint rotates around a single axis and has a limited range of motion.
    It is typically used for hinged joints like elbows or steering mechanisms.
    """

    CONTINUOUS = "continuous"
    """
    Similar to a revolute joint, but with no limits on its range of motion.
    It can rotate continuously around the axis. Ideal for wheels or continuous rotations.
    """

    PRISMATIC = "prismatic"
    """
    A prismatic joint allows for linear motion along a single axis.
    It is used for sliding or extending mechanisms, like a piston or linear actuator.
    """

    FLOATING = "floating"
    """
    A floating joint allows motion in all 6 degrees of freedom (3 translational, 3 rotational).
    It is often used to represent the base link of a free-floating robot.
    """

    PLANAR = "planar"
    """
    A planar joint allows motion in a plane perpendicular to the axis.
    It has 3 degrees of freedom: two translational and one rotational.
    """

    SPHERICAL = "spherical"
    """
    A spherical joint, also known as a ball joint, allows rotation around three axes.
    It has 3 rotational degrees of freedom but no translation.
    """

# %%
from urdfpy.urdf import Sphere

def scale_inertia(inertia: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Scale the 3x3 inertia matrix.
    
    :param inertia: 3x3 inertia matrix
    :param scale_factor: The factor by which to scale the model
    :return: Scaled 3x3 inertia matrix
    """
    # Inertia scales with the fifth power of linear dimensions
    return inertia * (scale_factor ** 5)

def scale_urdf_model(urdf_model: URDF, scale_factor: float) -> URDF:
    """
    Scale an entire URDF model by a given factor.
    
    :param urdf_model: The original URDF model
    :param scale_factor: The factor by which to scale the model
    :return: A new, scaled URDF model
    """
    
    links=[]
    for link in urdf_model.links:
        visuals = []
        collisions = []
        inertial = None        
        # Scale visuals
        for visual in link.visuals:
            scaled_visual = Visual(
                geometry=scale_geometry(visual.geometry, scale_factor),
                material=visual.material,
                origin=visual.origin * scale_factor
            )
            visuals.append(scaled_visual)
        
        # Scale collisions (if present)
        for collision in link.collisions:
            scaled_collision = Collision(
                name=collision.name,
                geometry=scale_geometry(collision.geometry, scale_factor),
                origin=visual.origin * scale_factor
            )
            collisions.append(scaled_collision)
        
        # Scale inertial properties
        if link.inertial:
            scaled_inertial = Inertial(
                mass=link.inertial.mass,
                origin=visual.origin * scale_factor,
                inertia=scale_inertia(link.inertial.inertia, scale_factor)
            )
            inertial = scaled_inertial
        
        links.append(
            Link(name=link.name, visuals=visuals, collisions=collisions, inertial=inertial)
        )
    
    # Scale joints
    joints = []
    for joint in urdf_model.joints:
        scaled_joint = Joint(
            name=joint.name,
            joint_type=joint.joint_type,
            parent=joint.parent,
            child=joint.child,
            axis=joint.axis,
            origin=joint.origin * scale_factor,
            limit=joint.limit,
            dynamics=joint.dynamics
        )
        joints.append(scaled_joint)
    return URDF(name=f"{urdf_model.name}_scaled", joints=joints, links=links)

def scale_geometry(geometry: Geometry, scale_factor: float) -> Geometry:
    if isinstance(geometry.box, Box):
        geometry.box = Box(size=np.array(geometry.box) * scale_factor)
    elif isinstance(geometry.cylinder, Cylinder):
        geometry.cylinder = Cylinder(radius=geometry.cylinder.radius * scale_factor, length=geometry.cylinder.length * scale_factor)
    elif isinstance(geometry.sphere, Sphere):
        geometry.cylinder = Sphere(radius=geometry.sphere.radius * scale_factor)        
    return geometry



# %%

class ShapeInertias:

    @staticmethod
    def get_uniform_box_inertia(mass: float, length_width_height_m: Tuple[float, float, float]) -> np.ndarray:
        """
        Compute the inertia of a uniform box
        :param mass: mass of the box
        :param size: size of the box (x, y, z)
        :return: mass and inertia
        """
        a, b, c = length_width_height_m
        return np.array([
            [1/12 * mass * (b**2 + c**2), 0, 0],
            [0, 1/12 * mass * (a**2 + c**2), 0],
            [0, 0, 1/12 * mass * (a**2 + b**2)]
        ])
        
    @staticmethod
    def get_uniform_cylinder_inertia(mass: float, radius: float, height: float) -> np.ndarray:
        """
        Compute the inertia of a uniform cylinder
        :param mass: mass of the cylinder
        :param radius: radius of the cylinder
        :param height: height of the cylinder
        :return: mass and inertia
        """
        # Moment of inertia for a solid cylinder
        Ix_Iy = (1/12) * mass * (3 * radius**2 + height**2)  # about diameter
        Iz = (1/2) * mass * radius**2  # about central axis
        
        return np.array([
            [Ix_Iy, 0.0, 0.0],
            [0.0, Ix_Iy, 0.0],
            [0.0, 0.0, Iz]
        ])


# %%
# IDS

CHASSIS_ID = 'chassis'


# %%

from urdfpy.urdf import Texture


def create_chassis_link(props: Chassis) -> Link:
    length_width_height_m = (props.length_m, props.width_m, props.height_m)
    geo = Geometry(box=Box(length_width_height_m))
    uniform_cuboid_inertia = ShapeInertias.get_uniform_box_inertia(props.mass_kg, length_width_height_m)

    return Link(
        visuals=[Visual(
            geometry=geo,
            material=Material('chassis_color', color=ColorsRgba.BLACK)
        )],
        collisions=[Collision(
            geometry=Geometry(box=Box(size=length_width_height_m)),
            name='chassis_collision'
        )],
        inertial=Inertial(mass=props.mass_kg, inertia=uniform_cuboid_inertia),
        name=CHASSIS_ID
    )

def create_wheel_link(
    name: str,
    props: Wheel
) -> Link:
    radius_m = props.dimeter_m / 2
    geo = Cylinder(radius_m, props.width_m)
    inertial_matrix =  ShapeInertias.get_uniform_cylinder_inertia(props.mass_kg, radius_m, props.width_m)
    return Link(
        visuals=[Visual(Geometry(cylinder=geo), Material('wheel_color', texture=Texture(filename=f"{project_root_dir}/textures/black-alloy-wheel.png")))],
        collisions=[Collision(geometry=Geometry(cylinder=geo), name='wheel_collision')],
        inertial=Inertial(props.mass_kg, inertia=inertial_matrix),
        name=name
    )
    
def create_axle_link(position_id: str, axle: AxleProperties) -> Link:
    # Create a simple cylindrical link for the axle
    axle_radius_m = axle.diameter_m / 2
    
    geo = create_uniform_cylinder_geometry(axle.length_m, axle_radius_m)
    return Link(
        name=f"{position_id}_axle",
        visuals=[Visual(
            geometry=geo,
            material=Material('axle_color', color=ColorsRgba.GREY)
        )],
        collisions=[Collision(
            geometry=geo,
            name=f"{position_id}_axle_collision"
        )],
        inertial=Inertial(
            mass=robot_properties.axle.mass_kg,
            inertia=ShapeInertias.get_uniform_cylinder_inertia(
                axle.mass_kg,
                axle_radius_m,
                axle.length_m
            )
        )
    )

def create_uniform_cylinder_geometry(length_m: float, radius_m: float) -> Geometry:
    return Geometry(cylinder=Cylinder(radius=radius_m, length=length_m))

def create_wheel_shock_mount_column_link(column_name: str, properties: WheelMount) -> Link:
   
    radius_m = properties.diameter_m / 2
    
    geo = create_uniform_cylinder_geometry(properties.length_m, radius_m)
    column_name = column_name + "_shock_mount_column"
    return Link(
        name=column_name,
        visuals=[Visual(
            geometry=geo,
            material=Material('shock_mount_column_color', color=ColorsRgba.GREY)
        )],
        collisions=[Collision(
            geometry=geo,
            name=f"{column_name}_collision"
        )],
        inertial=Inertial(
            mass=robot_properties.axle.mass_kg,
            inertia=ShapeInertias.get_uniform_cylinder_inertia(
                properties.mass_kg,
                radius_m,
                properties.length_m
            )
        ),
    )

# %%

chassis_link = create_chassis_link(robot_properties.chassis)

# %%

from enum import Enum
from typing import Generic, TypeVar
from pydantic.generics import GenericModel

T = TypeVar('T')

class WheelPosition(Enum):
    LEFT_FRONT = "left_front"
    RIGHT_FRONT = "right_front"
    LEFT_REAR = "left_rear"
    RIGHT_REAR = "right_rear"
    

class WheelDirectionsDetails(GenericModel, Generic[T]):
    left: T
    right: T


class WheelDetails(GenericModel, Generic[T]):
    front: WheelDirectionsDetails[T]
    rear: WheelDirectionsDetails[T]

# %%
wheel_ids = [WheelPosition.LEFT_FRONT, WheelPosition.RIGHT_FRONT, WheelPosition.LEFT_REAR, WheelPosition.RIGHT_REAR]


# %%
wheel_links = WheelDetails(
    front=WheelDirectionsDetails(
        left=create_wheel_link(WheelPosition.LEFT_FRONT.value, robot_properties.wheel),
        right=create_wheel_link(WheelPosition.RIGHT_FRONT.value, robot_properties.wheel)
    ),
    rear=WheelDirectionsDetails(
        left=create_wheel_link(WheelPosition.LEFT_REAR.value, robot_properties.wheel),
        right=create_wheel_link(WheelPosition.RIGHT_REAR.value, robot_properties.wheel)
    )
)

# %%
axle_links = WheelDetails(
    front=WheelDirectionsDetails(
        left=create_axle_link(WheelPosition.LEFT_FRONT.value, robot_properties.axle),
        right=create_axle_link(WheelPosition.RIGHT_FRONT.value, robot_properties.axle)
    ),
    rear=WheelDirectionsDetails(
        left=create_axle_link(WheelPosition.LEFT_REAR.value, robot_properties.axle),
        right=create_axle_link(WheelPosition.RIGHT_REAR.value, robot_properties.axle)
    )
)

# %%
front_steering_columns = WheelDirectionsDetails(
    left=create_wheel_shock_mount_column_link(WheelPosition.LEFT_FRONT.value, robot_properties.chassis.wheel_mounts),
    right=create_wheel_shock_mount_column_link(WheelPosition.RIGHT_FRONT.value, robot_properties.chassis.wheel_mounts)
)
    

# %%

# Try joining one wheel to one axle
from scripts.models import ControlledJoint, HardwareInterfaceType, TransmissionType
from urdfpy.urdf import Actuator, Transmission, TransmissionJoint


def create_wheel_axle_joint(wheel_id: str, axle_id: str) -> ControlledJoint:
    difference_between_wheel_and_axle = robot_properties.axle.length_m - robot_properties.wheel.width_m
    center_position_on_axle = difference_between_wheel_and_axle / 2 - robot_properties.axle.wheel_inset_m
    print(f"Creating joint from {axle_id} to {wheel_id}")
    joint_name = f"{wheel_id}_to_{axle_id}_joint"
    hardware_interface_type = HardwareInterfaceType.VELOCITY_JOINT.value
    return ControlledJoint(
            joint=Joint(
            name=joint_name,
            joint_type=JointType.CONTINUOUS.value,
            parent=axle_id,
            child=wheel_id,
            axis=[0, 0, 1], # Rotate around the z-axis
            origin=xyz_rpy_to_matrix([0, 0, center_position_on_axle, 0, 0, 0]),
            
        ),
        transmission=Transmission(
                name=f"{joint_name}_transmission",
                trans_type=TransmissionType.SIMPLE.value,
                actuators=[
                    Actuator(
                        name=f"{joint_name}_actuator", 
                        hardwareInterfaces=[hardware_interface_type],
                        mechanicalReduction=1
                        ),
                ],
                joints=[TransmissionJoint(name=joint_name, hardwareInterfaces=[hardware_interface_type])]
            )
    )

# %%
left_front_wheel_controllable_joint = create_wheel_axle_joint(wheel_links.front.left.name, axle_links.front.left.name)

# %%
wheel_axle_test_id = 'wheel_axle_test'
wheel_axel_test_urdf = URDF(
    name=wheel_axle_test_id,
    links=[wheel_links.front.left, axle_links.front.left],
    joints=[left_front_wheel_controllable_joint.joint]
)

# %%
# urdf.show()
# wheel_axel_test_urdf.animate(cfg_trajectory={
#     left_front_wheel_joint.value : [-np.pi / 4, np.pi / 4],

# })

# %%
import shutil
import subprocess

def render_urdf(urdf: URDF):
    output_file_path = f'{urdf.name}.urdf'
    urdf.save(output_file_path)
    BINARY_NAME = "urdf-viz"
    BINARY_PATH = shutil.which(BINARY_NAME)
    def run_binary(file_path: str):
        process = subprocess.Popen([BINARY_PATH, file_path]) # type: ignore

    run_binary(output_file_path)

# %%
# render_urdf(wheel_axel_test_urdf)

# %%
# Try joining one wheel to one axle
import math


def create_axle_column_joint(axle_id: str, column_id: str, axle: AxleProperties) -> Joint:
    return Joint(
        name=f"{axle_id}_to_{column_id}_joint",
        joint_type=JointType.FIXED.value,
        parent=column_id,
        child=axle_id,
        axis=[0, 1, 0], # Rotate around the z-axis
        origin=xyz_rpy_to_matrix([axle.length_m/2, 0, 0, 0, math.pi/2, 0,])
    )

# %%
left_front_axle_column_joint = create_axle_column_joint(axle_links.front.left.name, front_steering_columns.left.name, robot_properties.axle)

# %%
font_left_axle_link = axle_links.front.left
font_left_steering_link = front_steering_columns.left

# %%
vars(font_left_axle_link)

# %%
# render_urdf(
#     scale_urdf_model(URDF(
#         name="Axle column test",
#         links=[font_left_axle_link, font_left_steering_link],
#         joints=[left_front_axle_column_joint]
#     ) , 10
#     )
# )

# %%

urdf = URDF(
        name="Axle column test",
        links=[font_left_axle_link, font_left_steering_link],
        joints=[left_front_axle_column_joint]
    )

# %%
# urdf.show()
# urdf.animate(cfg_trajectory={
#     left_front_axle_column_joint.name : [-np.pi / 4, np.pi / 4],

# })
# render_urdf(
#     urdf
# )

# %%
urdf = URDF(
        name="Steering assembly test",
        links=[font_left_axle_link, font_left_steering_link, wheel_links.front.left],
        joints=[left_front_axle_column_joint, left_front_wheel_controllable_joint.joint],
        transmissions=[left_front_wheel_controllable_joint.transmission]
    )

# %%
# urdf.show()
# urdf.animate(cfg_trajectory={
#     left_front_wheel_joint.name : [-np.pi / 4, np.pi / 4],

# })

# %%

# Try joining one wheel to one axle
import math


def create_steering_column_chassis_joint(
    chassis_id: str, 
    column_id: str, 
    steering_column: SteeringColumn, 
    xyz_rpy_mount: Tuple[float, float, float, float, float, float]
) -> ControlledJoint:
    joint_name = f"{chassis_id}_to_{column_id}_joint"
    hardware_interface_type = HardwareInterfaceType.POSITION_JOINT.value
    return ControlledJoint(
        joint=Joint(
            name=joint_name,
            joint_type=JointType.REVOLUTE.value,
            parent=chassis_id,
            child=column_id,
            limit=JointLimit(
                lower=-math.radians(steering_column.max_angle_deg),
                upper=math.radians(steering_column.max_angle_deg),
                effort=steering_column.effort_Nm,
                velocity=steering_column.rads_per_sec
            ),
            axis=[0, 0, 1], # Rotate around the z-axis
            origin=xyz_rpy_to_matrix(xyz_rpy_mount)
        ),
        transmission=Transmission(
            name=f"{joint_name}_transmission",
            trans_type=TransmissionType.SIMPLE.value,
            actuators=[
                Actuator(
                    name=f"{joint_name}_actuator", 
                    hardwareInterfaces=[hardware_interface_type],
                    mechanicalReduction=1
                    ),
            ],
            joints=[TransmissionJoint(name=joint_name, hardwareInterfaces=[hardware_interface_type])]
        )
    )
    
def create_fixed_column_chassis_joint(chassis_id: str, column_id: str, xyz_rpy_mount: Tuple[float, float, float, float, float, float]) -> Joint:
    return Joint(
        name=f"{chassis_id}_to_{column_id}_joint",
        joint_type=JointType.FIXED.value,
        parent=chassis_id,
        child=column_id,
        origin=xyz_rpy_to_matrix(xyz_rpy_mount)
    )

# %%

# You can create a similar joint for the right front wheel
# front_right_assembly_xyz_r_mount = (
#     robot_properties.chassis.wheelbase / 2,
#     -robot_properties.chassis.wheel_x_offset_from_edge_m,  # Note the negative y value
#     robot_properties.chassis.wheel_z_base_offset_m,
#     math.pi/2
# )
front_wheel_x_m_distance_from_chassis_center = robot_properties.chassis.wheelbase / 2
wheel_z_value_from_chassis_center = -robot_properties.chassis.wheel_z_base_offset_m

front_left_assembly_xyz_r_mount = (
    front_wheel_x_m_distance_from_chassis_center, # x
    robot_properties.chassis.steering_column_y_center_offset_m, # y
    wheel_z_value_from_chassis_center, # z
    0, 0,math.pi/2
)

left_front_column_chassis_controllable_joint = create_steering_column_chassis_joint(chassis_link.name,  front_steering_columns.left.name, robot_properties.steering_column, front_left_assembly_xyz_r_mount)



# %%
# urdf = URDF(
#         name="Chassis with left steering assembly_test",
#         links=[font_left_axle_link, font_left_steering_link, wheel_links.front.left, chassis_link],
#         joints=[left_front_axle_column_joint, left_front_wheel_joint, left_front_column_chassis_joint]
#     )

# %%
# urdf.show()
# urdf.animate(cfg_trajectory={
#     left_front_wheel_joint.name : [-math.pi / 4, math.pi / 4],    
#     # left_front_column_chassis_joint:[-math.pi / 4, math.pi / 4],
# })
# render_urdf(
#     urdf
# )

# %%
front_right_axle_link = axle_links.front.right
front_right_steering_link = front_steering_columns.right
right_front_controllable_wheel_joint = create_wheel_axle_joint(wheel_links.front.right.name, axle_links.front.right.name)
right_front_axle_column_joint = create_axle_column_joint(axle_links.front.right.name, front_steering_columns.right.name, robot_properties.axle)
front_right_assembly_xyz_rpy_mount = (
    front_wheel_x_m_distance_from_chassis_center, # x
    -robot_properties.chassis.steering_column_y_center_offset_m, # y
    wheel_z_value_from_chassis_center, # z
    0,
    0,
    math.radians(270) 
)
right_front_column_chassis_controllable_joint = create_steering_column_chassis_joint(chassis_link.name,  front_steering_columns.right.name, robot_properties.steering_column, front_right_assembly_xyz_rpy_mount)

# %%
left_front_steering_assembly_links = (font_left_axle_link, font_left_steering_link, wheel_links.front.left)
left_front_joints = left_front_axle_column_joint, left_front_wheel_controllable_joint.joint, left_front_column_chassis_controllable_joint.joint

right_front_steering_assembly_links = (front_right_axle_link, front_right_steering_link, wheel_links.front.right)
right_front_joints = right_front_axle_column_joint, right_front_controllable_wheel_joint.joint, right_front_column_chassis_controllable_joint.joint

urdf = URDF(
        name="Chassis with left and right steering assembly_test",
        links=[*right_front_steering_assembly_links ,*left_front_steering_assembly_links, chassis_link],
        joints=[*right_front_joints, *left_front_joints],
        transmissions=[
            right_front_controllable_wheel_joint.transmission, 
            left_front_wheel_controllable_joint.transmission,
            left_front_column_chassis_controllable_joint.transmission,
            right_front_column_chassis_controllable_joint.transmission
        ]
    )

# %%
# urdf.show()
# urdf.animate(cfg_trajectory={
#     left_front_wheel_controllable_joint.joint.name : [-math.pi / 4, math.pi / 4],    
#     left_front_column_chassis_joint:[-math.pi / 4, math.pi / 4],
#     right_front_controllable_wheel_joint.joint.name : [-math.pi / 4, math.pi / 4],    
#     right_front_column_chassis_joint:[-math.pi / 4, math.pi / 4],
# })
# render_urdf(
#     urdf
# )

# %% [markdown]
# # Create rear assemblies

# %%
rear_wheel_x_m_distance_from_chassis_center = robot_properties.chassis.wheelbase / 2

# %%
rear_shock_columns = WheelDirectionsDetails(
    left=create_wheel_shock_mount_column_link(WheelPosition.LEFT_REAR.value, robot_properties.chassis.wheel_mounts),
    right=create_wheel_shock_mount_column_link(WheelPosition.RIGHT_REAR.value, robot_properties.chassis.wheel_mounts),
)

# %%

rear_axel_controllable_joints = WheelDirectionsDetails(
    left=create_wheel_axle_joint(wheel_links.rear.left.name, axle_links.rear.left.name),
    right=create_wheel_axle_joint(wheel_links.rear.right.name, axle_links.rear.right.name)
)
rear_axel_controllable_joints.left.joint.parent

# %%
rear_axle_column_joint = WheelDirectionsDetails(
    left=create_axle_column_joint(axle_links.rear.left.name, rear_shock_columns.left.name, robot_properties.axle),
    right=create_axle_column_joint(axle_links.rear.right.name, rear_shock_columns.right.name, robot_properties.axle)
)

# %%
rear_left_assembly_xyz_rpy_mount = (
    -front_wheel_x_m_distance_from_chassis_center, # x
    robot_properties.chassis.steering_column_y_center_offset_m, # y
    wheel_z_value_from_chassis_center, # z
    0,
    0, 
    math.radians(90)
)

rear_right_assembly_xyz_rpy_mount = (
    -front_wheel_x_m_distance_from_chassis_center, # x
    -robot_properties.chassis.steering_column_y_center_offset_m, # y
    wheel_z_value_from_chassis_center, # z
    0,
    0, 
    math.radians(270)
)

# %%
rear_chassis_joints = WheelDirectionsDetails(
    left=create_fixed_column_chassis_joint(CHASSIS_ID, rear_shock_columns.left.name, rear_left_assembly_xyz_rpy_mount),
    right=create_fixed_column_chassis_joint(CHASSIS_ID, rear_shock_columns.right.name, rear_right_assembly_xyz_rpy_mount)
)    

# %%
right_rear_steering_assembly_links = (wheel_links.rear.right, axle_links.rear.right, rear_shock_columns.right)
left_rear_steering_assembly_links = (wheel_links.rear.left, axle_links.rear.left, rear_shock_columns.left)
right_rear_joints = (rear_chassis_joints.right, rear_axel_controllable_joints.right.joint, rear_axle_column_joint.right)
left_rear_joints = (rear_chassis_joints.left, rear_axel_controllable_joints.left.joint, rear_axle_column_joint.left)

# %%

links = [*right_front_steering_assembly_links ,*left_front_steering_assembly_links, *right_rear_steering_assembly_links, *left_rear_steering_assembly_links,chassis_link]
for link in links:
    print(link.name)


# %%
front_joints=  [*right_front_joints, *left_front_joints]
for joint in front_joints:
    print(joint.name)

# %%
rear_joints = [*right_rear_joints, *left_rear_joints]
for joint in rear_joints:
    print(joint)

# %%
joints = [ *front_joints, *rear_joints]
for joint in joints:
    print(joint.name)

# %%
urdf = URDF(
        name="Chassis with left and right steering assembly_test",
        links=links,
        joints=joints,
        transmissions=[
            right_front_controllable_wheel_joint.transmission, 
            left_front_wheel_controllable_joint.transmission,
            left_front_column_chassis_controllable_joint.transmission,
            right_front_column_chassis_controllable_joint.transmission,
            rear_axel_controllable_joints.right.transmission,
            rear_axel_controllable_joints.left.transmission,
        ]
    )

# %%
# urdf.show()
urdf.animate(cfg_trajectory={
    left_front_wheel_controllable_joint.joint.name : [-math.pi / 4, math.pi / 4],    
    left_front_column_chassis_controllable_joint.joint:[-math.pi / 4, math.pi / 4],
    right_front_controllable_wheel_joint.joint.name : [-math.pi / 4, math.pi / 4],    
    right_front_column_chassis_controllable_joint.joint:[-math.pi / 4, math.pi / 4],
    rear_axel_controllable_joints.left.joint.name : [-math.pi / 4, math.pi / 4],
    rear_axel_controllable_joints.right.joint.name:[-math.pi / 4, math.pi / 4], 
})
# render_urdf(
#     urdf
# )
urdf.save("boticar.urdf")

# %%
from dataclasses import dataclass


@dataclass
class WheelAssembly:
    wheel: Link
    axle: Link
    steering_column: Link
    wheel_axle_joint: Joint
    axle_column_joint: Joint
    column_chassis_joint: Joint

# %%



