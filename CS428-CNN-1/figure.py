
import re
import pandas as pd
import matplotlib.pyplot as plt

# Input log data as a string
log_data = """
  1/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.3750 - loss: 1.5748/opt/anaconda3/envs/CS428/lib/python3.10/contextlib.py:153: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self.gen.throw(typ, value, traceback)
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 650us/step - accuracy: 0.3750 - loss: 1.5748 - val_accuracy: 0.1568 - val_loss: 15.7448
Epoch 3/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 14ms/step - accuracy: 0.3157 - loss: 1.9632 - val_accuracy: 0.2936 - val_loss: 2.0123
Epoch 4/100
2024-11-26 12:31:06.518783: I tensorflow/core/framework/local_rendezvous.cc:405] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 642us/step - accuracy: 0.3750 - loss: 2.0286 - val_accuracy: 0.2968 - val_loss: 2.0111
Epoch 5/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 15ms/step - accuracy: 0.3997 - loss: 1.6773 - val_accuracy: 0.5448 - val_loss: 1.3701
Epoch 6/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 647us/step - accuracy: 0.6250 - loss: 1.3094 - val_accuracy: 0.5456 - val_loss: 1.3683
Epoch 7/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 15ms/step - accuracy: 0.4499 - loss: 1.5147 - val_accuracy: 0.5952 - val_loss: 1.2114
Epoch 8/100
2024-11-26 12:31:21.134000: I tensorflow/core/framework/local_rendezvous.cc:405] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 672us/step - accuracy: 0.7500 - loss: 1.1269 - val_accuracy: 0.5896 - val_loss: 1.2101
Epoch 9/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.4914 - loss: 1.4457 - val_accuracy: 0.6336 - val_loss: 1.0603
Epoch 10/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 679us/step - accuracy: 0.7500 - loss: 0.8775 - val_accuracy: 0.6352 - val_loss: 1.0643
Epoch 11/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.5349 - loss: 1.3239 - val_accuracy: 0.7688 - val_loss: 0.7127
Epoch 12/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 659us/step - accuracy: 0.6250 - loss: 1.0937 - val_accuracy: 0.7864 - val_loss: 0.6957
Epoch 13/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 15ms/step - accuracy: 0.5619 - loss: 1.1996 - val_accuracy: 0.6576 - val_loss: 0.9388
Epoch 14/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 658us/step - accuracy: 0.6250 - loss: 0.7749 - val_accuracy: 0.6552 - val_loss: 0.9374
Epoch 15/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.6084 - loss: 1.1217 - val_accuracy: 0.6192 - val_loss: 0.9948
Epoch 16/100
2024-11-26 12:31:51.804715: I tensorflow/core/framework/local_rendezvous.cc:405] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 672us/step - accuracy: 0.2500 - loss: 1.8948 - val_accuracy: 0.6192 - val_loss: 0.9893
Epoch 17/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.6127 - loss: 1.0735 - val_accuracy: 0.8160 - val_loss: 0.5441
Epoch 18/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 672us/step - accuracy: 0.7500 - loss: 1.0125 - val_accuracy: 0.8192 - val_loss: 0.5393
Epoch 19/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.6210 - loss: 1.0354 - val_accuracy: 0.7904 - val_loss: 0.6227
Epoch 20/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 680us/step - accuracy: 0.8750 - loss: 0.7191 - val_accuracy: 0.7824 - val_loss: 0.6296
Epoch 21/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 8s 16ms/step - accuracy: 0.6547 - loss: 0.9917 - val_accuracy: 0.7456 - val_loss: 0.7139
Epoch 22/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 683us/step - accuracy: 0.6250 - loss: 0.7605 - val_accuracy: 0.7440 - val_loss: 0.7170
Epoch 23/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 8s 16ms/step - accuracy: 0.6605 - loss: 0.9371 - val_accuracy: 0.6312 - val_loss: 1.0164
Epoch 24/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 670us/step - accuracy: 0.7500 - loss: 0.8405 - val_accuracy: 0.6272 - val_loss: 1.0266
Epoch 25/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 8s 16ms/step - accuracy: 0.6871 - loss: 0.9284 - val_accuracy: 0.8256 - val_loss: 0.4809
Epoch 26/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 674us/step - accuracy: 0.6250 - loss: 0.7968 - val_accuracy: 0.8296 - val_loss: 0.4760
Epoch 27/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 8s 16ms/step - accuracy: 0.6834 - loss: 0.8686 - val_accuracy: 0.7992 - val_loss: 0.5336
Epoch 28/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 670us/step - accuracy: 0.5000 - loss: 1.3258 - val_accuracy: 0.7992 - val_loss: 0.5333
Epoch 29/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 8s 16ms/step - accuracy: 0.7186 - loss: 0.8180 - val_accuracy: 0.8336 - val_loss: 0.4695
Epoch 30/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 677us/step - accuracy: 0.8750 - loss: 0.5554 - val_accuracy: 0.8456 - val_loss: 0.4574
Epoch 31/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 8s 16ms/step - accuracy: 0.7139 - loss: 0.7862 - val_accuracy: 0.8808 - val_loss: 0.3660
Epoch 32/100
2024-11-26 12:32:54.892268: I tensorflow/core/framework/local_rendezvous.cc:405] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 686us/step - accuracy: 0.8750 - loss: 0.4512 - val_accuracy: 0.8760 - val_loss: 0.3755
Epoch 33/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 8s 16ms/step - accuracy: 0.7477 - loss: 0.7387 - val_accuracy: 0.9016 - val_loss: 0.3046
Epoch 34/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 674us/step - accuracy: 0.3750 - loss: 1.4271 - val_accuracy: 0.9024 - val_loss: 0.3065
Epoch 35/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 8s 16ms/step - accuracy: 0.7424 - loss: 0.7618 - val_accuracy: 0.8552 - val_loss: 0.4049
Epoch 36/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 661us/step - accuracy: 0.6250 - loss: 1.1824 - val_accuracy: 0.8528 - val_loss: 0.4104
Epoch 37/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.7401 - loss: 0.7330 - val_accuracy: 0.8976 - val_loss: 0.3690
Epoch 38/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 661us/step - accuracy: 0.7500 - loss: 0.8056 - val_accuracy: 0.9016 - val_loss: 0.3598
Epoch 39/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.7466 - loss: 0.6958 - val_accuracy: 0.8360 - val_loss: 0.4402
Epoch 40/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 653us/step - accuracy: 0.6250 - loss: 0.7331 - val_accuracy: 0.8368 - val_loss: 0.4379
Epoch 41/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.7579 - loss: 0.6728 - val_accuracy: 0.9256 - val_loss: 0.2436
Epoch 42/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 664us/step - accuracy: 0.6250 - loss: 0.5560 - val_accuracy: 0.9280 - val_loss: 0.2362
Epoch 43/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.7399 - loss: 0.7113 - val_accuracy: 0.9096 - val_loss: 0.2840
Epoch 44/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 656us/step - accuracy: 0.8750 - loss: 0.3608 - val_accuracy: 0.9088 - val_loss: 0.2860
Epoch 45/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.7681 - loss: 0.7017 - val_accuracy: 0.9176 - val_loss: 0.2379
Epoch 46/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 652us/step - accuracy: 0.5000 - loss: 1.1214 - val_accuracy: 0.9192 - val_loss: 0.2378
Epoch 47/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.7594 - loss: 0.7021 - val_accuracy: 0.9208 - val_loss: 0.2608
Epoch 48/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 672us/step - accuracy: 0.7500 - loss: 0.9076 - val_accuracy: 0.9216 - val_loss: 0.2580
Epoch 49/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.7584 - loss: 0.6720 - val_accuracy: 0.9440 - val_loss: 0.2031
Epoch 50/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 670us/step - accuracy: 0.8750 - loss: 0.2470 - val_accuracy: 0.9440 - val_loss: 0.2040
Epoch 51/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.7778 - loss: 0.6300 - val_accuracy: 0.7808 - val_loss: 0.7468
Epoch 52/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 665us/step - accuracy: 1.0000 - loss: 0.2266 - val_accuracy: 0.7832 - val_loss: 0.7238
Epoch 53/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.7833 - loss: 0.6087 - val_accuracy: 0.9528 - val_loss: 0.1933
Epoch 54/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 660us/step - accuracy: 0.8750 - loss: 0.7799 - val_accuracy: 0.9528 - val_loss: 0.1922
Epoch 55/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.7816 - loss: 0.6359 - val_accuracy: 0.9304 - val_loss: 0.2107
Epoch 56/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 664us/step - accuracy: 0.8750 - loss: 0.6494 - val_accuracy: 0.9312 - val_loss: 0.2108
Epoch 57/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.8081 - loss: 0.5421 - val_accuracy: 0.9368 - val_loss: 0.2047
Epoch 58/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 658us/step - accuracy: 1.0000 - loss: 0.1523 - val_accuracy: 0.9376 - val_loss: 0.2045
Epoch 59/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.8063 - loss: 0.5462 - val_accuracy: 0.9536 - val_loss: 0.1503
Epoch 60/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 658us/step - accuracy: 1.0000 - loss: 0.2624 - val_accuracy: 0.9560 - val_loss: 0.1498
Epoch 61/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.8151 - loss: 0.5430 - val_accuracy: 0.9512 - val_loss: 0.1755
Epoch 62/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 661us/step - accuracy: 0.3750 - loss: 0.9416 - val_accuracy: 0.9504 - val_loss: 0.1775
Epoch 63/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.8203 - loss: 0.5341 - val_accuracy: 0.9576 - val_loss: 0.1545
Epoch 64/100
2024-11-26 12:34:57.704230: I tensorflow/core/framework/local_rendezvous.cc:405] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 667us/step - accuracy: 0.6250 - loss: 0.8716 - val_accuracy: 0.9584 - val_loss: 0.1533
Epoch 65/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.7997 - loss: 0.5818 - val_accuracy: 0.9520 - val_loss: 0.1581
Epoch 66/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 659us/step - accuracy: 0.7500 - loss: 0.9959 - val_accuracy: 0.9520 - val_loss: 0.1589
Epoch 67/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.8165 - loss: 0.5425 - val_accuracy: 0.9488 - val_loss: 0.1657
Epoch 68/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 668us/step - accuracy: 0.8750 - loss: 0.3501 - val_accuracy: 0.9472 - val_loss: 0.1661
Epoch 69/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.8263 - loss: 0.4930 - val_accuracy: 0.9352 - val_loss: 0.2250
Epoch 70/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 664us/step - accuracy: 0.8750 - loss: 0.3375 - val_accuracy: 0.9352 - val_loss: 0.2271
Epoch 71/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.8174 - loss: 0.5156 - val_accuracy: 0.9568 - val_loss: 0.1567
Epoch 72/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 664us/step - accuracy: 1.0000 - loss: 0.2126 - val_accuracy: 0.9568 - val_loss: 0.1570
Epoch 73/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.8224 - loss: 0.5305 - val_accuracy: 0.9688 - val_loss: 0.1313
Epoch 74/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 651us/step - accuracy: 0.7500 - loss: 0.8931 - val_accuracy: 0.9680 - val_loss: 0.1314
Epoch 75/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.8145 - loss: 0.5529 - val_accuracy: 0.9488 - val_loss: 0.1478
Epoch 76/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 691us/step - accuracy: 1.0000 - loss: 0.2436 - val_accuracy: 0.9488 - val_loss: 0.1483
Epoch 77/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.8326 - loss: 0.5199 - val_accuracy: 0.9648 - val_loss: 0.1329
Epoch 78/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 655us/step - accuracy: 0.8750 - loss: 0.3395 - val_accuracy: 0.9640 - val_loss: 0.1334
Epoch 79/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 8s 16ms/step - accuracy: 0.8190 - loss: 0.5153 - val_accuracy: 0.9672 - val_loss: 0.1265
Epoch 80/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 656us/step - accuracy: 0.8750 - loss: 0.2829 - val_accuracy: 0.9672 - val_loss: 0.1272
Epoch 81/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 8s 18ms/step - accuracy: 0.8357 - loss: 0.4869 - val_accuracy: 0.9664 - val_loss: 0.1287
Epoch 82/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 730us/step - accuracy: 0.6250 - loss: 0.5770 - val_accuracy: 0.9664 - val_loss: 0.1286
Epoch 83/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 8s 16ms/step - accuracy: 0.8262 - loss: 0.5121 - val_accuracy: 0.9688 - val_loss: 0.1170
Epoch 84/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 656us/step - accuracy: 0.7500 - loss: 0.3540 - val_accuracy: 0.9688 - val_loss: 0.1175
Epoch 85/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 12s 26ms/step - accuracy: 0.8301 - loss: 0.5086 - val_accuracy: 0.9688 - val_loss: 0.1382
Epoch 86/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.8750 - loss: 0.6570 - val_accuracy: 0.9680 - val_loss: 0.1381
Epoch 87/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 9s 20ms/step - accuracy: 0.8219 - loss: 0.5026 - val_accuracy: 0.9544 - val_loss: 0.1580
Epoch 88/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 870us/step - accuracy: 1.0000 - loss: 0.1249 - val_accuracy: 0.9544 - val_loss: 0.1573
Epoch 89/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 8s 16ms/step - accuracy: 0.8321 - loss: 0.4985 - val_accuracy: 0.9680 - val_loss: 0.1199
Epoch 90/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 637us/step - accuracy: 0.8750 - loss: 0.3494 - val_accuracy: 0.9688 - val_loss: 0.1192
Epoch 91/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 15ms/step - accuracy: 0.8502 - loss: 0.4497 - val_accuracy: 0.9680 - val_loss: 0.1107
Epoch 92/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 638us/step - accuracy: 0.6250 - loss: 0.6037 - val_accuracy: 0.9680 - val_loss: 0.1113
Epoch 93/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 15ms/step - accuracy: 0.8501 - loss: 0.4282 - val_accuracy: 0.9624 - val_loss: 0.1245
Epoch 94/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 652us/step - accuracy: 0.8750 - loss: 0.2052 - val_accuracy: 0.9624 - val_loss: 0.1253
Epoch 95/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.8416 - loss: 0.4520 - val_accuracy: 0.9688 - val_loss: 0.1153
Epoch 96/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 654us/step - accuracy: 1.0000 - loss: 0.0937 - val_accuracy: 0.9688 - val_loss: 0.1147
Epoch 97/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 15ms/step - accuracy: 0.8479 - loss: 0.4766 - val_accuracy: 0.9680 - val_loss: 0.1154
Epoch 98/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 652us/step - accuracy: 0.8750 - loss: 0.2457 - val_accuracy: 0.9672 - val_loss: 0.1166
Epoch 99/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 7s 16ms/step - accuracy: 0.8617 - loss: 0.4060 - val_accuracy: 0.9568 - val_loss: 0.1274
Epoch 100/100
468/468 ━━━━━━━━━━━━━━━━━━━━ 0s 659us/step - accuracy: 1.0000 - loss: 0.1970 - val_accuracy: 0.9568 - val_loss: 0.1254
[INFO] evaluating network...
157/157 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step 
"""

# Regular expression to match the metrics for each epoch
epoch_pattern = re.compile(
    r"Epoch (\d+)/\d+\n.*?accuracy: ([0-9.]+) - loss: ([0-9.]+) - val_accuracy: ([0-9.]+) - val_loss: ([0-9.]+)"
)

# Extract results
results = []
for match in epoch_pattern.finditer(log_data):
    epoch = int(match.group(1))
    accuracy = float(match.group(2))
    loss = float(match.group(3))
    val_accuracy = float(match.group(4))
    val_loss = float(match.group(5))
    results.append([epoch, accuracy, loss, val_accuracy, val_loss])

# Convert results to a Pandas DataFrame
columns = ["Epoch", "Accuracy", "Loss", "Val Accuracy", "Val Loss"]
df = pd.DataFrame(results, columns=columns)

# Save the results to a CSV file
df.to_csv("training_results_1.csv", index=False)

# Display the DataFrame
print(df)

# Plotting Accuracy and Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(df["Epoch"], df["Accuracy"], label="Training Accuracy", marker="o", color="blue")
plt.plot(df["Epoch"], df["Val Accuracy"], label="Validation Accuracy", marker="o", color="green")

# Add titles and labels
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")

# Save the graph to a file
plt.savefig("accuracy_over_epochs_1.png")
plt.show()