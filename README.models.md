# Wake Word Models Training Configuration

This repository contains trained wake word detection models with their respective training configurations.

*Nota*: Para melhorar, poderia tentar usar um treinar um com o piper e com a frase Clãriss, com brs e assim tal como fiz nos outros modelos.

# New models 28/07/2025
> Agora há um problema qualquer no notebook personal de treino em que há um problema com o pyarrow advindo da biblioteca `datasets` (`AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'`). A solução que encontrei foi correr estes comandos antes de abrir o jupyter notebook de treino:
> ```sh
> apt-get update && apt-get install -y build-essential
> sudo apt install nvidia-cuda-toolkit -y
> echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
> source ~/.bashrc
> ```
> 31/07/2025
> Aparentemente isto tbm não é suficiente, não consigo perceber muito bem o que se passa, desta vez reparei que o que fiz desde uma altura que não dava para a altura que atualmente dá, foi fazer um `sudo apt full-upgrade -y` e dar shutdown de todos os kernels através do botão na tree que indica `Shut Down All`. Tecnicamente também fiz um `Run Selected Cell And All Below` instead of a `Run All` no jupyter notebook.


## New - Hey Clarisse Models

| Model Name        | Training Samples | Training Steps | False Activation Penalty | Validation Samples | Accuracy  | Recall | False Positives per Hour | Notes                     |
| ----------------- | ---------------- | -------------- | ------------------------ | ------------------ | --------- | ------ | ------------------------ | ------------------------- |
| hey_cledeess-v4.0 | 40000            | 30000          | 5000                     | 500                | 0.71      | 0.42   | 0.17                     | English GPU training      |
| Hey_Clãriss-3.3   | 40000            | 30000          | 3000                     | 2000               | 0.78      | 0.56   | 0.18                     | Hey_Clãriss               |
| Hey_Clãriss-3.2   | 40000            | 100000         | 5000                     | 2000               | 0.73      | 0.46   | 0.09                     | Hey_Clãriss (in training) |
| Hey_Clãriss-3.2   | 35000            | 50000          | 4000                     | 2000               | 0.73      | 0.46   | 0.09                     | Hey_Clãriss               |
| Hey_Clãriss-3.1   | 35000            | 25000          | 3000                     | 2000               | 0.76      | 0.52   | 0.17                     | Hey_Clãriss               |
| Hey_Clãriss-2.7   | 35000            | 25000          | 3000                     | 2000               | 0.75      | 0.49   | 0.00                     | Hey_Clãriss               |
| Hey_Clãriss-2.6   | 50000            | 20000          | 3000                     | 2000               | 0.70      | 0.41   | 0.09                     | Hey_Clãriss               |
| Hey_Clãriss-2.5   | 35000            | 20000          | 3000                     | 1000               | 0.73      | 0.47   | 0.00                     | Hey_Clãriss               |
| Hey_Clãriss-2.4   | 25000            | 20000          | 3000                     | 1000               | *(76.86%) | -      | *(3.85%)                 | Hey_Clãriss               |
| Hey_Clãriss-2.3   | 10000            | 20000          | (default)                | 1000               | 0.73      | 0.46   | 0.44                     | Hey_Clãriss               |
| hey_cledeess-2.2  | 40000            | 20000          | 3500                     | 500                | *(73.14%) | -      | *(1.92%)                 | Google Colab (simple)     |
| hey_cledeess-2.1  | 40000            | 20000          | 5000                     | 500                | *(75.62%) | -      | *(1.92%)                 | Google Colab (simple)     |
| hey_cledees-2.0   | 40000            | 20000          | 1500                     | 500                | *(70.66%) | -      | *(3.85%)                 | Google Colab (simple)     |


## New - Olá Clarisse Models

| Model Name       | Training Samples | Training Steps | False Activation Penalty | Validation Samples | Accuracy | Recall | False Positives per Hour | Notes                                 |
| ---------------- | ---------------- | -------------- | ------------------------ | ------------------ | -------- | ------ | ------------------------ | ------------------------------------- |
| olá_clãriss-2.2  | 35000            | 20000          | 3000                     | 1000               | 0.78     | 0.56   | 0.00                     | Ólá Clãriss; Olá Clãriss              |
| olá_cledeess-2.1 | 40000            | 20000          | 3000                     | 500                | *??      | -      | *??                      | Google Colab (simple) "ólá_cledeess!" |
| olá_cledeess-2.0 | 40000            | 20000          | 5000                     | 500                | -        | -      | -                        | Google Colab (simple)                 |


---


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