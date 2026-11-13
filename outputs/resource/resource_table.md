| Model | Feature dim | Sparse | Params | Val Acc@0.5 | Val Macro-F1@0.5 | AUC | Acc@thr_acc | Macro-F1@thr_f1 | Seed std | Train s | Infer s | Peak MB | OOV | Hardware |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| roberta (avg of 3 seeds) | 768 | – | 124647170 | 0.9591 | 0.9194 | 0.9806 | 0.9625 | 0.9273 | 0.0043 | 113.1 | 7.715 | 2434 | – | CPU: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD (16 threads); GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| tfidf_svm | 50000 | True | 150009 | 0.9290 | 0.8476 | 0.9502 | 0.9296 | 0.8571 | 0.0000 | 2.8 | 0.004 | 150 | – | CPU: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD (16 threads) |
| tfidf_cnb | 10000 | True | 20000 | 0.9102 | 0.8385 | 0.9529 | 0.9263 | 0.8571 | 0.0000 | 1.2 | 0.004 | 155 | – | CPU: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD (16 threads) |
| bilstm | 300 | False | 2876010 | 0.9303 | 0.8524 | 0.9470 | 0.9303 | 0.8524 | 0.0050 | 121.6 | 4.574 | 1112 | 0.050 | CPU: x86_64 (9 threads) |
| tfidf_lr | 50000 | True | 50001 | 0.9236 | 0.8451 | 0.9504 | 0.9276 | 0.8512 | 0.0000 | 1.5 | 0.003 | 122 | – | CPU: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD (16 threads) |
| glove_lr | 300 | False | 301 | 0.8539 | 0.7623 | 0.9040 | 0.8995 | 0.7884 | 0.0000 | 0.1 | 0.001 | 351 | 0.050 | CPU: x86_64 (9 threads) |