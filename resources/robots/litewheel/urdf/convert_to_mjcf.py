"""Convert litewheel_rl.urdf to a MuJoCo MJCF with visual meshes."""

from pathlib import Path
import struct
import tempfile
import xml.etree.ElementTree as ET

import mujoco


directory = Path(__file__).resolve().parent
source = directory / "litewheel_rl.urdf"
target = directory / "litewheel_rl.xml"

robot = ET.parse(source)
mesh_directory = directory.parent / "meshes" / "mjcf"
mesh_directory.mkdir(exist_ok=True)
base_mesh = (directory.parent / "meshes" / "base_link.STL").read_bytes()
face_count = struct.unpack_from("<I", base_mesh, 80)[0]
assert len(base_mesh) == 84 + face_count * 50
split = 185000
for name, first, count in (("base_link_part1.stl", 0, split), ("base_link_part2.stl", split, face_count - split)):
    start = 84 + first * 50
    (mesh_directory / name).write_bytes(base_mesh[:80] + struct.pack("<I", count) + base_mesh[start:start + count * 50])
base_visual = robot.find(".//link[@name='Base']/visual")
base_visual.find("geometry/mesh").attrib["filename"] = "../meshes/mjcf/base_link_part1.stl"
second_visual = ET.fromstring(ET.tostring(base_visual))
second_visual.find("geometry/mesh").attrib["filename"] = "../meshes/mjcf/base_link_part2.stl"
robot.find(".//link[@name='Base']").append(second_visual)
extension = ET.SubElement(robot.getroot(), "mujoco")
ET.SubElement(extension, "compiler", discardvisual="false", fusestatic="false")

with tempfile.NamedTemporaryFile(suffix=".urdf", dir=directory, delete=False) as handle:
    temporary = Path(handle.name)

try:
    robot.write(temporary, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(temporary))
    mujoco.mj_saveLastXML(str(target), model)
finally:
    temporary.unlink()

tree = ET.parse(target)
worldbody = tree.getroot().find("worldbody")
base = worldbody.find("body")
base.insert(0, ET.Element("freejoint", name="floating_base"))
for joint in tree.findall(".//joint"):
    if joint.attrib.get("name", "").endswith("WHEEL_MOTOR"):
        joint.attrib.pop("range", None)
        joint.attrib["limited"] = "false"
for geom in tree.findall(".//geom"):
    if geom.attrib.get("contype") != "0":
        geom.attrib["group"] = "3"
actuators = ET.SubElement(tree.getroot(), "actuator")
for urdf_joint in robot.findall(".//joint"):
    name = urdf_joint.attrib["name"]
    effort = urdf_joint.find("limit").attrib["effort"]
    ET.SubElement(actuators, "motor", name=f"{name}_motor", joint=name,
                  ctrllimited="true", ctrlrange=f"-{effort} {effort}")
ET.indent(tree)
tree.write(target, encoding="utf-8", xml_declaration=True)

model = mujoco.MjModel.from_xml_path(str(target))
print(f"{target}: {model.nbody} bodies, {model.njnt} joints, {model.nmesh} meshes")
