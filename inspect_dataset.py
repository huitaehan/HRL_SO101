from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

repo_id = "huitaehan/so101_demo"
print(f"Fetching metadata for {repo_id}...")

meta = LeRobotDatasetMetadata(repo_id)

print("\n=== Dataset Summary ===")
print(f"Total episodes: {meta.total_episodes}")
print(f"Total frames: {meta.total_frames}")
print(f"FPS: {meta.fps}")

print("\n=== Features ===")
for name, ft in meta.features.items():
    print(f"  - {name}: dtype={ft.get('dtype')}, shape={ft.get('shape')}")

if hasattr(meta, "tasks") and not getattr(meta.tasks, "empty", True):
    print(f"\nTasks:\n{meta.tasks}")
