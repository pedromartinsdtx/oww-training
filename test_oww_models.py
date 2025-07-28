import glob

from openwakeword.model import Model
from openwakeword.utils import bulk_predict

WW_MODELS_FOLDER = "models-ww"
CLARISSE_MODELS = f"{WW_MODELS_FOLDER}/clarisse"
HEY_CLARISSE_MODELS = f"{WW_MODELS_FOLDER}/hey-clarisse"
OLA_CLARISSE_MODELS = f"{WW_MODELS_FOLDER}/ola-clarisse"
PARA_MODELS = f"{WW_MODELS_FOLDER}/para"

ACTIVATION_THRESHOLD = 0.5

model = Model()
# model.predict_clip("path/to/wav/file")

# audio_folder_paths = ["samples/clarisse"]
# audio_folder_paths = ["samples/olá-clarisse"]
audio_folder_paths = ["samples/hei-clarisse"]

audio_file_paths = []
for folder in audio_folder_paths:
    audio_file_paths.extend(glob.glob(f"{folder}/**/*.wav", recursive=True))

wakeword_models_paths = [
    # f"{CLARISSE_MODELS}/Clarisse_v-piper.onnx",
    # f"{CLARISSE_MODELS}/Clarisse_v1.2-piper.onnx",
    # f"{CLARISSE_MODELS}/Clarisse_v2_piper.onnx",
    # f"{CLARISSE_MODELS}/Clarisse_v2.5_piper.onnx",
    # f"{CLARISSE_MODELS}/Clarisse_v3_piper.onnx",
    # f"{CLARISSE_MODELS}/Clarisse_v4_piper.onnx",
    # f"{CLARISSE_MODELS}/Clarisse_v5_piper.onnx",
    # f"{CLARISSE_MODELS}/Clarisse_v6_piper.onnx",
    # f"{CLARISSE_MODELS}/CLEDEESSS_v5.onnx",
    # f"{CLARISSE_MODELS}/CLEDEESSS_v6.onnx",
    # f"{CLARISSE_MODELS}/cledeesss_v7.onnx",
    #
    # f"{OLA_CLARISSE_MODELS}/Olá_Clãriss-v1-piper.onnx",
    # f"{OLA_CLARISSE_MODELS}/Ólá_Clãriss-v2-piper.onnx",
    # f"{OLA_CLARISSE_MODELS}/holá_cleddeess.onnx",
    # f"{OLA_CLARISSE_MODELS}/olá_cleddeess.onnx",
    # f"{OLA_CLARISSE_MODELS}/olá_cleddeess-v2.onnx",
    # f"{OLA_CLARISSE_MODELS}/olá_cledeess-v3.onnx",
    # f"{OLA_CLARISSE_MODELS}/olá_cledeess-v4.onnx",
    #
    # f"{HEY_CLARISSE_MODELS}/eeii_cleddeess.onnx",
    # f"{HEY_CLARISSE_MODELS}/eeii_cleddeess_v2.onnx",
    f"{HEY_CLARISSE_MODELS}/Hey_Clariss_v1_piper.onnx",
    f"{HEY_CLARISSE_MODELS}/Hey_Clariss_v1.2_piper.onnx",
    f"{HEY_CLARISSE_MODELS}/Hey_Clariss_v2_piper.onnx",
    f"{HEY_CLARISSE_MODELS}/hey_cledees-2.0.onnx",
    f"{HEY_CLARISSE_MODELS}/hey_cledeess-2.1.onnx",
    #
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

false_audio_file_paths = []
false_positive_audio_folder_paths = ["samples/false"]

for folder in false_positive_audio_folder_paths:
    false_audio_file_paths.extend(glob.glob(f"{folder}/**/*.wav", recursive=True))

false_results = bulk_predict(
    file_paths=false_audio_file_paths,
    wakeword_models=wakeword_models_paths,
    ncpu=6,
    inference_framework="onnx",
)

total_false_files_processed = 0
total_false_files_activated = 0
false_model_activation_counts = {}
false_file_max_scores = {}

for file_path, segment_scores_list in false_results.items():
    total_false_files_processed += 1
    activated_in_file = False
    activation_details = {}
    max_scores = {}

    for segment_scores in segment_scores_list:
        for model_name, score in segment_scores.items():
            if model_name not in false_model_activation_counts:
                false_model_activation_counts[model_name] = 0
            if model_name not in max_scores or score > max_scores[model_name]:
                max_scores[model_name] = score
            if score > ACTIVATION_THRESHOLD:
                activated_in_file = True
                if (
                    model_name not in activation_details
                    or score > activation_details[model_name]
                ):
                    activation_details[model_name] = score

    false_file_max_scores[file_path] = max_scores

    if activated_in_file:
        total_false_files_activated += 1
        for model_name in activation_details.keys():
            false_model_activation_counts[model_name] += 1
    # print("  Max scores for all models:")
    # for model_name, max_score in max_scores.items():
    #     print(f"    {model_name}: {max_score:.4f}")

print("\n--- False Positive Final Report ---")
print(f"Total Files Processed: {total_false_files_processed}")
print(
    f"Total Files Activated: {total_false_files_activated} ({(total_false_files_activated / total_false_files_processed) * 100 if total_false_files_processed else 0:.2f}%)"
)
print(
    f"Total Files Not Activated: {total_false_files_processed - total_false_files_activated} ({((total_false_files_processed - total_false_files_activated) / total_false_files_processed) * 100 if total_false_files_processed else 0:.2f}%)"
)
print("\nModel Activation Counts (false files activated by each model):")
sorted_false_model_counts = sorted(
    false_model_activation_counts.items(),
    key=lambda item: (item[1] / total_false_files_processed) * 100
    if total_false_files_processed
    else 0,
    reverse=True,
)

for model_name, count in sorted_false_model_counts:
    percentage = (
        (count / total_false_files_processed) * 100
        if total_false_files_processed
        else 0
    )
    print(f"  {model_name}: {count} ({percentage:.2f}%)")
