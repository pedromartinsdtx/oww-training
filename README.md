# Useful commands

```sh
find / -name "libcudnn_adv.so.9" 2>/dev/null
echo 'export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

cp /opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn_adv.so.9 /opt/conda/share/jupyter/kernels/python3/.
```


Useful command for its notebook
```sh
grep -E "(=== Training combination|Final Model Accuracy|Final Model Recall|Final Model False Positives|Saving ONNX|N_samples)" output.txt
```





# Requirements
```sh
sudo apt install -y portaudio19-dev pulseaudio pulseaudio-utils ffmpeg
```
