# Wake Word Models Training Configuration

This repository contains trained wake word detection models with their respective training configurations.

## Model Training Parameters

| Model Name | Training Samples | Training Steps | False Activation Penalty | Target Accuracy | Target Recall | Validation Samples | Score Percentage |
|------------|------------------|----------------|--------------------------|-----------------|---------------|-------------------|------------------|
| eeii_cleddeess-v2 | 5050 | 19000 | 1500 | - | - | 100 | - |
| eeii_cleddeess | 1000 | 10000 | 1500 | - | - | 100 | - |
| olá_cleddeess-v2 | 2700 | 10000 | 1500 | - | - | 270 | - |
| olá_cleddeess | 1000 | 10000 | 1500 | - | - | 100 | - |
| holá_cleddeess | 1000 | 10000 | 1500 | - | - | 100 | - |
| cledeesss_v7 | 6350 | 10000 | 1500 | - | - | 500 | - |
| CLEDEESSS_v6 | 8250 | 11000 | 1500 | - | - | 500 | - |
| CLEDEESSS_v5 | - | - | - | - | - | 500 | - |
| CLEDEESSS_v4 | - | - | - | - | - | 500 | - |
| Cleh-DEE-sse_v3 | 8550 | 13300 | 1400 | - | - | - | - |
| Clarisse_v-piper | 10000 | 50000 | 1500 | 0.7 | 0.4 | 2000 | - | (A lot of spanish voices without tugao)
| Clarisse_v1.2-piper | 10000 | 50000 | 1500 | 0.7 | 0.4 | 2000 | - | (A lot of spanish voices with tugao)
| Clarisse_v2_piper | 10000 | 50000 | 1500 | 0.7 | 0.4 | - | - | (Just Tugao e Rita voices)
| Clarisse_v2.5_piper | - | 50000 | 1500 | - | - | - | - | (Just Tugao e Rita voices)
| Clarisse_v3_piper | 10000 | 50000 | 1500 | - | - | 5000 | - | (Tugao, Rita, tirei o "Clarisse, ")
| Clarisse_v4_piper | 5000 | 50000 | 1500 | - | - | 200 | - | (Tugao, Rita, tirei o "Clarisse, ")
| Clarisse_v5_piper | 5000 | 10000 | 1500 | - | - | 200 | - | (Tugao, Espanhol, italiano) (Model Accuracy: 0.6675; Model Recall: 0.335; Model False Positives per Hour: 0.0)
| Clarisse_v6_piper | 10000 | 20000 | 1500 | - | - | 200 | - | (Tugao, Rita, Espanhol, Italiano)

