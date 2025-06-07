import glob

from openwakeword.model import Model
from openwakeword.utils import bulk_predict

WW_MODELS_FOLDER = "models-ww"
ACTIVATION_THRESHOLD = 0.5

model = Model()
# model.predict_clip("path/to/wav/file")

audio_folder_paths = ["samples/clarisse"]
# audio_folder_paths = ["samples/olá-clarisse"]
audio_folder_paths = ["samples/hei-clarisse"]

audio_file_paths = []
for folder in audio_folder_paths:
    audio_file_paths.extend(glob.glob(f"{folder}/**/*.wav", recursive=True))

wakeword_models_paths = [
    # f"{WW_MODELS_FOLDER}/Clarisse_v-piper.onnx",
    # f"{WW_MODELS_FOLDER}/Clarisse_v1.2-piper.onnx",
    # f"{WW_MODELS_FOLDER}/Clarisse_v2_piper.onnx",
    # f"{WW_MODELS_FOLDER}/Clarisse_v2.5_piper.onnx",
    # f"{WW_MODELS_FOLDER}/CLEDEESSS_v5.onnx",
    # f"{WW_MODELS_FOLDER}/CLEDEESSS_v6.onnx",
    # "models-ww/eeii_cleddeess.onnx",
    # f"{WW_MODELS_FOLDER}/cledeesss_v7.onnx",
    # "models-ww/holá_cleddeess.onnx",
    # "models-ww/olá_cleddeess.onnx",
    # "models-ww/olá_cleddeess-v2.onnx",
    "models-ww/eeii_cleddeess.onnx",
    "models-ww/eeii_cleddeess_v2.onnx",
]
# wakeword_models_paths = glob.glob(f"{WW_MODELS_FOLDER}/*.onnx")

# Get audio data containing 16-bit 16khz PCM audio data from a file, microphone, network stream, etc.
# For the best efficiency and latency, audio frames should be multiples of 80 ms, with longer frames
# increasing overall efficiency at the cost of detection latency
results = bulk_predict(
    file_paths=audio_file_paths,
    wakeword_models=wakeword_models_paths,
    ncpu=6,
    inference_framework="onnx",
)

print("Results:")
total_files_processed = 0
total_files_activated = 0
model_activation_counts = {}
file_max_scores = {}

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
            if score > ACTIVATION_THRESHOLD:
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
print(
    f"Total Files Activated: {total_files_activated} ({(total_files_activated / total_files_processed) * 100:.2f}%)"
)
print(
    f"Total Files Not Activated: {total_files_processed - total_files_activated} ({((total_files_processed - total_files_activated) / total_files_processed) * 100:.2f}%)"
)
print("\nModel Activation Counts (files activated by each model):")
sorted_model_counts = sorted(
    model_activation_counts.items(),
    key=lambda item: (item[1] / total_files_processed) * 100,
    reverse=True,
)

for model_name, count in sorted_model_counts:
    percentage = (count / total_files_processed) * 100
    print(f"  {model_name}: {count} ({percentage:.2f}%)")
