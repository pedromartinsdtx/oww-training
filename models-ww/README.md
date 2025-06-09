# Wake Word Models Training Configuration

This repository contains trained wake word detection models with their respective training configurations.

## Hey Clarisse Models

| Model Name          | Training Samples | Training Steps | False Activation Penalty | Validation Samples | Accuracy        | Recall | False Positives per Hour | Notes                                   |
| ------------------- | ---------------- | -------------- | ------------------------ | ------------------ | --------------- | ------ | ------------------------ | --------------------------------------- |
| cledeesss_v7        | 6350             | 10000          | 1500                     | 500                | *80.76%         | -      | -                        |                                         |
| CLEDEESSS_v6        | 8250             | 11000          | 1500                     | 500                | *82.68%         | -      | -                        |                                         |
| CLEDEESSS_v5        | -                | -              | -                        | 500                | *83.63%         | -      | -                        |                                         |
| CLEDEESSS_v4        | -                | -              | -                        | 500                | *85.90%         | -      | -                        |                                         |
| Cleh-DEE-sse_v3     | 8550             | 13300          | 1400                     | -                  | -               | -      | -                        |                                         |
| Clarisse_v-piper    | 10000            | 50000          | 1500                     | 2000               | -               | -      | -                        | (A lot of spanish voices without tugao) |
| Clarisse_v1.2-piper | 10000            | 50000          | 1500                     | 2000               | -               | -      | -                        | (A lot of spanish voices with tugao)    |
| Clarisse_v2_piper   | 10000            | 50000          | 1500                     | -                  | *47.67%         | -      | -                        | (Just Tugao e Rita voices)              |
| Clarisse_v2.5_piper | -                | 50000          | 1500                     | -                  | *55.08%         | -      | -                        | (Just Tugao e Rita voices)              |
| Clarisse_v3_piper   | 10000            | 50000          | 1500                     | 5000               | *(baixa)        | -      | -                        | (Tugao, Rita, tirei o "Clarisse, ")     |
| Clarisse_v4_piper   | 5000             | 50000          | 1500                     | 200                | *(baixa)        | -      | -                        | (Tugao, Rita, tirei o "Clarisse, ")     |
| Clarisse_v5_piper   | 5000             | 10000          | 1500                     | 200                | 0.6675 *(baixa) | 0.335  | 0.0                      | (Tugao, Espanhol, italiano)             |
| Clarisse_v6_piper   | 10000            | 20000          | 1500                     | 500                | 0.757 *(48%)    | 0.516  | 0.0                      | (Tugao, Rita, Espanhol, Italiano)       |

## Olá Clarisse Models

| Model Name       | Training Samples | Training Steps | False Activation Penalty | Validation Samples | Accuracy | Recall | False Positives per Hour | Notes |
| ---------------- | ---------------- | -------------- | ------------------------ | ------------------ | -------- | ------ | ------------------------ | ----- |
| olá_cleddeess-v2 | 2700             | 10000          | 1500                     | 270                | *61%     | -      | -                        |       |
| olá_cleddeess    | 1000             | 10000          | 1500                     | 100                | *78%     | -      | -                        |       |
| holá_cleddeess   | 1000             | 10000          | 1500                     | 100                | *40%     | -      | -                        |       |

## Hey Clarisse Models

| Model Name             | Training Samples | Training Steps | False Activation Penalty | Validation Samples | Accuracy | Recall | False Positives per Hour | Notes                             |
| ---------------------- | ---------------- | -------------- | ------------------------ | ------------------ | -------- | ------ | ------------------------ | --------------------------------- |
| hey_Clariss_v2_piper   | 10000            | 20000          | 1500                     | 500                | 0.74     | 0.48   | 0.7                      | Clãriss (PTs, BRs, Espanhol)      |
| Hey_Clariss_v1_piper   | 20000            | 30000          | 1500                     | 500                | 0.80     | 0.60   | 0.5                      | Clariss (PTs, Espanhol, Italiano) |
| Hey_Clariss_v1.2_piper | 30000            | 40000          | 1500                     | 500                | 0.81     | 0.62   | 1.07                     | Clariss (PTs, Espanhol, Italiano) |
| eeii_cleddeess-v2      | 5050             | 19000          | 1500                     | -                  | *62%     | -      | -                        | Colab                             |
| eeii_cleddeess         | 1000             | 10000          | 1500                     | -                  | *59%     | -      | -                        | Colab                             |

