import json
import os


def save_data(folder: str, file_name: str, data: dict[str, float]) -> str:
    path = os.path.join(folder, file_name)
    try:
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False)
            return "200"
    except Exception as e:
        return str(e)
