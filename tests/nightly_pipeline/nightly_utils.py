import os


def human_readable(size):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024


def get_onnx_and_qpc_size(dir):
    total_size = 0
    for root, dirs, files in os.walk(dir):
        for name in files:
            file_path = os.path.join(root, name)
            if not os.path.islink(file_path):  # avoid counting symlinks
                total_size += os.path.getsize(file_path)
    print(f"Total size of {dir}: {total_size} bytes")
    return human_readable(total_size)
