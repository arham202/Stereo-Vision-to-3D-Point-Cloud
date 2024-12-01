import open3d as o3d

ply_file = r"C:\Projects\Stereo Vision to 3D Point Cloud\output.ply"   # Replace with your PLY file path
pcd = o3d.io.read_point_cloud(ply_file)

print(pcd)

o3d.visualization.draw_geometries([pcd])
