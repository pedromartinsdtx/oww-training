import glob

from openwakeword.model import Model
from openwakeword.utils import bulk_predict

model = Model()
# model.predict_clip("path/to/wav/file")

# folder_paths = ["samples/samples_edge_pt_augmented"]
folder_paths = ["samples/gemini"]

file_paths = []
for folder in folder_paths:
    file_paths.extend(glob.glob(f"{folder}/*.wav"))

results = bulk_predict(
    file_paths=file_paths,
    # wakeword_model_paths=[
    wakeword_models=[
        "models-ww/CLEDEESSS_v5.onnx",
        "models-ww/CLEDEESSS_v6.onnx",
        # "models-ww/CLEDEESSS_v7.tflite",
    ],
    ncpu=2,
    inference_framework="onnx",
)

print("Results:")
total_files_processed = 0
total_files_activated = 0
model_activation_counts = {}
file_max_scores = {}  # {file_path: {model_name: max_score}}

for file_path, segment_scores_list in results.items():
    total_files_processed += 1
    activated_in_file = False
    activation_details = {}
    max_scores = {}  # max score for each model in this file

    for segment_scores in segment_scores_list:
        for model_name, score in segment_scores.items():
            if model_name not in model_activation_counts:
                model_activation_counts[model_name] = 0
            # Track max score for each model
            if model_name not in max_scores or score > max_scores[model_name]:
                max_scores[model_name] = score
            if score > 0.5:
                activated_in_file = True
                if (
                    model_name not in activation_details
                    or score > activation_details[model_name]
                ):
                    activation_details[model_name] = score

    file_max_scores[file_path] = max_scores

    if activated_in_file:
        total_files_activated += 1
        print(f"File: {file_path} - ACTIVATED")
        # for model_name, max_score in activation_details.items():
        #     print(f"  Model: {model_name}, Max Activation Score: {max_score:.4f}")
        for model_name in activation_details.keys():
            model_activation_counts[model_name] += 1
    else:
        print(f"File: {file_path} - NOT ACTIVATED")
    # Always print max scores for each model
    print("  Max scores for all models:")
    for model_name, max_score in max_scores.items():
        print(f"    {model_name}: {max_score:.4f}")

print("\n--- Final Report ---")
print(f"Total Files Processed: {total_files_processed}")
print(f"Total Files Activated: {total_files_activated}")
print(f"Total Files Not Activated: {total_files_processed - total_files_activated}")
print("\nModel Activation Counts (files activated by each model):")
for model_name, count in model_activation_counts.items():
    print(f"  {model_name}: {count}")
