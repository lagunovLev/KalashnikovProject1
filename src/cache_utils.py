import os
import json
import hashlib
import inspect
from pathlib import Path

def get_file_hash(file_path):
    """Вычисляет MD5 хеш файла."""
    if not os.path.exists(file_path):
        return None
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_function_hash(func):
    """Вычисляет MD5 хеш исходного кода функции."""
    source_code = inspect.getsource(func)
    return hashlib.md5(source_code.encode()).hexdigest()

class PipelineState:
    def __init__(self, state_file):
        self.state_file = Path(state_file)
        self.state = self._load()

    def _load(self):
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {}

    def save(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=4)

    def is_step_needed(self, step_name, func, input_files, config_params):
        """Проверяет, нужно ли запускать шаг пайплайна."""
        func_hash = get_function_hash(func)
        input_hashes = {str(f): get_file_hash(f) for f in input_files}
        
        step_state = self.state.get(step_name)
        if not step_state:
            return True

        if step_state.get("func_hash") != func_hash:
            return True
        
        if step_state.get("input_hashes") != input_hashes:
            return True
            
        if step_state.get("config_params") != config_params:
            return True

        # Проверка существования выходных файлов, если они указаны в конфиге
        # (Это добавим позже в логику main.py)
        
        return False

    def update_step(self, step_name, func, input_files, config_params):
        """Обновляет состояние шага после успешного выполнения."""
        self.state[step_name] = {
            "func_hash": get_function_hash(func),
            "input_hashes": {str(f): get_file_hash(f) for f in input_files},
            "config_params": config_params
        }
        self.save()
