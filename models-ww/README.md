# Wake Word Models Training Configuration

This repository contains trained wake word detection models with their respective training configurations.

*Note*: The accuracy values with an asterisk `(*)` are derived from my own validation script, not the models' testing data.

## Clarisse Models

*Nota*: Para melhorar, poderia tentar usar um treinar um com o piper e com a frase Clãriss, com brs e assim tal como fiz nos outros modelos.

| Model Name          | Training Samples | Training Steps | False Activation Penalty | Validation Samples | Accuracy        | Recall | False Positives per Hour | Notes                             |
| ------------------- | ---------------- | -------------- | ------------------------ | ------------------ | --------------- | ------ | ------------------------ | --------------------------------- |
| Clarisse_v6_piper   | 10000            | 20000          | 1500                     | 500                | 0.757 *(48%)    | 0.516  | 0.0                      | (Tugao, Rita, Espanhol, Italiano) |
| cledeesss_v7        | 6350             | 10000          | 1500                     | 500                | *80.76%         | ?      | ?                        | Colab                             |
| CLEDEESSS_v6        | 8250             | 11000          | 1500                     | 500                | *82.68%         | ?      | ?                        | Colab                             |
| CLEDEESSS_v5        | ?                | ?              | ?                        | 500                | *83.63%         | ?      | ?                        | Colab                             |
| CLEDEESSS_v4        | ?                | ?              | ?                        | 500                | *85.90%         | ?      | ?                        | Colab                             |
| Clarisse_v5_piper   | 5000             | 10000          | 1500                     | 200                | 0.6675 *(baixa) | 0.335  | 0.0                      | (Tugao, Espanhol, italiano)       |
| Clarisse_v2.5_piper | -                | 50000          | 1500                     | ?                  | *55.08%         | ?      | ?                        | (Just Tugao e Rita voices)        |
| Clarisse_v2_piper   | 10000            | 50000          | 1500                     | ?                  | *47.67%         | ?      | ?                        | (Just Tugao e Rita voices)        |

## Olá Clarisse Models

| Model Name           | Training Samples | Training Steps | False Activation Penalty | Validation Samples | Accuracy    | Recall | False Positives per Hour | Notes                                         |
| -------------------- | ---------------- | -------------- | ------------------------ | ------------------ | ----------- | ------ | ------------------------ | --------------------------------------------- |
| Ólá_Clãriss-v2_piper | 20000            | 30000          | 1500                     | 1000               | 0.79 *(95%) | 0.58   | 0.7                      | Ólá Clãriss(?) (PTs, BRs, Espanhol)           |
| olá_cledeess-v4      | 5000             | 15000          | 1500                     | 500                | *95%        | -      | -                        | Colab                                         |
| olá_cledeess-v3      | 1000             | 10000          | 1500                     | 100                | *90%        | ?      | ?                        | Colab                                         |
| Olá_Clãriss-v1_piper | 20000            | 30000          | 1500                     | 1000               | 0.84 *(88%) | 0.68   | 0.35                     | Olá Clãriss(?) (PTs, BRs, Espanhol, Italiano) |
| olá_cleddeess-v2     | 2700             | 10000          | 1500                     | 270                | *61%        | ?      | ?                        | Colab                                         |
| olá_cleddeess        | 1000             | 10000          | 1500                     | 100                | *78%        | ?      | ?                        | Colab                                         |
| holá_cleddeess       | 1000             | 10000          | 1500                     | 100                | *40%        | ?      | ?                        | Colab                                         |

## Hey Clarisse Models

| Model Name             | Training Samples | Training Steps | False Activation Penalty | Validation Samples | Accuracy | Recall | False Positives per Hour | Notes                             |
| ---------------------- | ---------------- | -------------- | ------------------------ | ------------------ | -------- | ------ | ------------------------ | --------------------------------- |
| hey_Clariss_v2_piper   | 10000            | 20000          | 1500                     | 500                | 0.74     | 0.48   | 0.7                      | Clãriss (PTs, BRs, Espanhol)      |
| Hey_Clariss_v1.2_piper | 30000            | 40000          | 1500                     | 500                | 0.81     | 0.62   | 1.07                     | Clariss (PTs, Espanhol, Italiano) |
| Hey_Clariss_v1_piper   | 20000            | 30000          | 1500                     | 500                | 0.80     | 0.60   | 0.5                      | Clariss (PTs, Espanhol, Italiano) |
| eeii_cleddeess-v2      | 5050             | 19000          | 1500                     | ?                  | *62%     | ?      | ?                        | Colab                             |
| eeii_cleddeess         | 1000             | 10000          | 1500                     | ?                  | *59%     | ?      | ?                        | Colab                             |


## Pára Models

...
| Model Name | Training Samples | Training Steps | False Activation Penalty | Validation Samples | Accuracy | Recall | False Positives per Hour | Notes |
| ---------- | ---------------- | -------------- | ------------------------ | ------------------ | -------- | ------ | ------------------------ | ----- |