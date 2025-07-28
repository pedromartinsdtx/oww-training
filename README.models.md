# Wake Word Models Training Configuration

This repository contains trained wake word detection models with their respective training configurations.

*Nota*: Para melhorar, poderia tentar usar um treinar um com o piper e com a frase Clãriss, com brs e assim tal como fiz nos outros modelos.

## New models 28/07/2025


## New - Hey Clarisse Models

| Model Name       | Training Samples | Training Steps | False Activation Penalty | Validation Samples | Accuracy  | Recall | False Positives per Hour | Notes                 |
| ---------------- | ---------------- | -------------- | ------------------------ | ------------------ | --------- | ------ | ------------------------ | --------------------- |
| hey_cledeess-2.1 | 40000            | 20000          | 5000                     | 500                | *(78.07%) | -      | *(1.92%)                 | Google Colab (simple) |
| hey_cledees-2.0  | 40000            | 20000          | 1500                     | 500                | *(73.80%) | -      | *(3.85%)                 | Google Colab (simple) |

## New - Olá Clarisse Models

| Model Name       | Training Samples | Training Steps | False Activation Penalty | Validation Samples | Accuracy | Recall | False Positives per Hour | Notes                 |
| ---------------- | ---------------- | -------------- | ------------------------ | ------------------ | -------- | ------ | ------------------------ | --------------------- |
| olá_cledeess-2.0 | 40000            | 20000          | 5000                     | 500                | -        | -      | -                        | Google Colab (simple) |


## Olá Clarisse Models

| Model Name           | Training Samples | Training Steps | False Activation Penalty | Validation Samples | Accuracy    | Recall | False Positives per Hour | Notes                                         |
| -------------------- | ---------------- | -------------- | ------------------------ | ------------------ | ----------- | ------ | ------------------------ | --------------------------------------------- |
| -> olá_cledeess-v4   | 5000             | 15000          | 1500                     | 500                | *95%        | -      | -                        | Colab                                         |
| Ólá_Clãriss-v2-piper | 20000            | 30000          | 1500                     | 1000               | 0.79 *(95%) | 0.58   | 0.7                      | Ólá Clãriss(?) (PTs, BRs, Espanhol)           |
| olá_cledeess-v3      | 1000             | 10000          | 1500                     | 100                | *90%        | ?      | ?                        | Colab                                         |
| Olá_Clãriss-v1-piper | 20000            | 30000          | 1500                     | 1000               | 0.84 *(88%) | 0.68   | 0.35                     | Olá Clãriss(?) (PTs, BRs, Espanhol, Italiano) |
| olá_cleddeess        | 1000             | 10000          | 1500                     | 100                | *78%        | ?      | ?                        | Colab                                         |
| holá_cleddeess       | 1000             | 10000          | 1500                     | 100                | *40%        | ?      | ?                        | Colab                                         |

## Hey Clarisse Models

| Model Name              | Training Samples | Training Steps | False Activation Penalty | Validation Samples | Accuracy | Recall | False Positives per Hour | Notes                             |
| ----------------------- | ---------------- | -------------- | ------------------------ | ------------------ | -------- | ------ | ------------------------ | --------------------------------- |
| -> hey_Clariss_v2_piper | 10000            | 20000          | 1500                     | 500                | 0.74     | 0.48   | 0.7    *(9.62%)          | Clãriss (PTs, BRs, Espanhol)      |
| Hey_Clariss_v1.2_piper  | 30000            | 40000          | 1500                     | 500                | 0.81     | 0.62   | 1.07                     | Clariss (PTs, Espanhol, Italiano) |
| Hey_Clariss_v1_piper    | 20000            | 30000          | 1500                     | 500                | 0.80     | 0.60   | 0.5                      | Clariss (PTs, Espanhol, Italiano) |


> *Note*: The accuracy values with an asterisk `*` are derived from my own [validation script](oww-training/test_oww_models.py), not the models' testing data.
> 
> *Note*: The `->` indicates that these are the currently used models.
> 
> Based from the original list in [openwakeword/models-ww/README.md](oww-training/models-ww/README.md).
>
> Also there is this repository with some trained models and their specs just to use as reference: https://github.com/fwartner/home-assistant-wakewords-collection/blob/main/en/hal/README.md