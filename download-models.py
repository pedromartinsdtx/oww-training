import os
import openwakeword as oww

models_path = os.path.join(os.path.dirname(oww.__file__), "resources", "models")
os.makedirs(models_path, exist_ok=True)
oww.utils.download_models()
print("Models downloaded successfully:", os.listdir(models_path))
