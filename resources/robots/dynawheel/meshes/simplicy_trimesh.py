import trimesh

mesh = trimesh.load_mesh('lf_wheel.STL')

# 正确写法：
simplified = mesh.simplify_quadric_decimation(face_count=10000)  # 明确指定参数名

print(f"原始面数: {len(mesh.faces)}")
print(f"简化后面数: {len(simplified.faces)}")

# 保存为二进制格式
simplified.export('lf_wheel_simple.STL', file_type='stl')