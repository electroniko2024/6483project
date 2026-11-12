| Model | Feature dim | Sparse | Params | Val Acc@0.5 | Val Macro-F1@0.5 | AUC | Acc@thr_acc | Macro-F1@thr_f1 | Seed std | Train s | Infer s | Peak MB | OOV | Hardware |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tfidf_svm | 50000 | True | 150009 | 0.9290 | 0.8476 | 0.9502 | 0.9296 | 0.8571 | 0.0000 | 2.8 | 0.004 | 150 | – | CPU: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD (16 threads) |
| tfidf_cnb | 10000 | True | 20000 | 0.9102 | 0.8385 | 0.9529 | 0.9263 | 0.8571 | 0.0000 | 1.2 | 0.004 | 155 | – | CPU: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD (16 threads) |
| tfidf_lr | 50000 | True | 50001 | 0.9236 | 0.8451 | 0.9504 | 0.9276 | 0.8512 | 0.0000 | 1.5 | 0.003 | 122 | – | CPU: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD (16 threads) |