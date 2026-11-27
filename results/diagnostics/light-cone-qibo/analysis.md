# Diagnóstico del cono de luz de la QCNN, arquitectura `qibo` (generado)

Generado por `experiments/diagnose_light_cone.py`. Propiedad del circuito, independiente de los datos.

Pesos entrenables: 42 (forma [2, 21]); apariciones parametrizadas: 102; CNOT: 3; qubit de lectura: 3.

## Codificación angle (40 entradas y pesos aleatorios)

- Características sin influencia en la salida: [] (cambio máximo por característica: [0.6573, 0.4851, 0.8555, 0.5645]).
- Pesos `[capa, k]` sin influencia: [[0, 'pool5'], [1, 'pool5']].
- Apariciones con contribución al gradiente: 99 de 102.
- Grado trigonométrico efectivo por peso `capa,k`: {'0,conv0': [4], '0,conv1': [4], '0,conv2': [4], '0,conv3': [4], '0,conv4': [4], '0,conv5': [4], '0,conv6': [4], '0,conv7': [4], '0,conv8': [4], '0,conv9': [4], '0,conv10': [4], '0,conv11': [4], '0,conv12': [4], '0,conv13': [4], '0,conv14': [4], '0,pool0': [4], '0,pool1': [4], '0,pool2': [4], '0,pool3': [2], '0,pool4': [2], '0,pool5': [0], '1,conv0': [1], '1,conv1': [1], '1,conv2': [1], '1,conv3': [1], '1,conv4': [1], '1,conv5': [1], '1,conv6': [1], '1,conv7': [1], '1,conv8': [1], '1,conv9': [1], '1,conv10': [1], '1,conv11': [1], '1,conv12': [1], '1,conv13': [1], '1,conv14': [1], '1,pool0': [2], '1,pool1': [2], '1,pool2': [2], '1,pool3': [1], '1,pool4': [1], '1,pool5': [0]}.
- Circuitos por muestra: parameter-shift 205 (actual) frente a 199 (solo apariciones vivas); secuencial 246 (actual) frente a 238 (grado efectivo).
- Diferencia máxima simulador rápido frente a Qibo: 1.1e-15.

| Aparición | Contribución máxima |
|---|---|
| 000-layer0-rx0-conv0 | 0.3083 |
| 001-layer0-ry0-conv1 | 0.3494 |
| 002-layer0-rz0-conv2 | 0.4134 |
| 003-layer0-rx1-conv3 | 0.4580 |
| 004-layer0-ry1-conv4 | 0.3200 |
| 005-layer0-rz1-conv5 | 0.4569 |
| 006-layer0-rzz01-conv6 | 0.4655 |
| 007-layer0-ryy01-conv7 | 0.4822 |
| 008-layer0-rxx01-conv8 | 0.3625 |
| 009-layer0-rx0-conv9 | 0.3188 |
| 010-layer0-ry0-conv10 | 0.4217 |
| 011-layer0-rz0-conv11 | 0.3539 |
| 012-layer0-rx1-conv12 | 0.4684 |
| 013-layer0-ry1-conv13 | 0.3772 |
| 014-layer0-rz1-conv14 | 0.4469 |
| 015-layer0-rx2-conv0 | 0.3887 |
| 016-layer0-ry2-conv1 | 0.4838 |
| 017-layer0-rz2-conv2 | 0.3417 |
| 018-layer0-rx3-conv3 | 0.4128 |
| 019-layer0-ry3-conv4 | 0.3325 |
| 020-layer0-rz3-conv5 | 0.3197 |
| 021-layer0-rzz23-conv6 | 0.4112 |
| 022-layer0-ryy23-conv7 | 0.3461 |
| 023-layer0-rxx23-conv8 | 0.5689 |
| 024-layer0-rx2-conv9 | 0.5630 |
| 025-layer0-ry2-conv10 | 0.3404 |
| 026-layer0-rz2-conv11 | 0.3656 |
| 027-layer0-rx3-conv12 | 0.4087 |
| 028-layer0-ry3-conv13 | 0.3091 |
| 029-layer0-rz3-conv14 | 0.2768 |
| 030-layer0-rx1-conv0 | 0.4165 |
| 031-layer0-ry1-conv1 | 0.4300 |
| 032-layer0-rz1-conv2 | 0.4480 |
| 033-layer0-rx2-conv3 | 0.3009 |
| 034-layer0-ry2-conv4 | 0.3796 |
| 035-layer0-rz2-conv5 | 0.5658 |
| 036-layer0-rzz12-conv6 | 0.4834 |
| 037-layer0-ryy12-conv7 | 0.3286 |
| 038-layer0-rxx12-conv8 | 0.5058 |
| 039-layer0-rx1-conv9 | 0.2862 |
| 040-layer0-ry1-conv10 | 0.2613 |
| 041-layer0-rz1-conv11 | 0.3404 |
| 042-layer0-rx2-conv12 | 0.5688 |
| 043-layer0-ry2-conv13 | 0.5036 |
| 044-layer0-rz2-conv14 | 0.4455 |
| 045-layer0-rx3-conv0 | 0.3682 |
| 046-layer0-ry3-conv1 | 0.2614 |
| 047-layer0-rz3-conv2 | 0.4279 |
| 048-layer0-rx0-conv3 | 0.3830 |
| 049-layer0-ry0-conv4 | 0.3431 |
| 050-layer0-rz0-conv5 | 0.4345 |
| 051-layer0-rzz30-conv6 | 0.4692 |
| 052-layer0-ryy30-conv7 | 0.6225 |
| 053-layer0-rxx30-conv8 | 0.5367 |
| 054-layer0-rx3-conv9 | 0.6156 |
| 055-layer0-ry3-conv10 | 0.3800 |
| 056-layer0-rz3-conv11 | 0.5831 |
| 057-layer0-rx0-conv12 | 0.3613 |
| 058-layer0-ry0-conv13 | 0.3360 |
| 059-layer0-rz0-conv14 | 0.4128 |
| 060-layer0-rx2-pool0 | 0.3819 |
| 061-layer0-ry2-pool1 | 0.5175 |
| 062-layer0-rz2-pool2 | 0.5718 |
| 063-layer0-rx0-pool3 | 0.4110 |
| 064-layer0-ry0-pool4 | 0.3901 |
| 065-layer0-rz0-pool5 | 0.0000 |
| 066-layer0-rz2--pool2 | 0.6076 |
| 067-layer0-ry2--pool1 | 0.4611 |
| 068-layer0-rx2--pool0 | 0.4622 |
| 069-layer0-rx3-pool0 | 0.3508 |
| 070-layer0-ry3-pool1 | 0.4025 |
| 071-layer0-rz3-pool2 | 0.6798 |
| 072-layer0-rx1-pool3 | 0.2687 |
| 073-layer0-ry1-pool4 | 0.4277 |
| 074-layer0-rz1-pool5 | 0.0000 |
| 075-layer0-rz3--pool2 | 0.5784 |
| 076-layer0-ry3--pool1 | 0.3925 |
| 077-layer0-rx3--pool0 | 0.4584 |
| 078-layer1-rx2-conv0 | 0.4622 |
| 079-layer1-ry2-conv1 | 0.4253 |
| 080-layer1-rz2-conv2 | 0.4788 |
| 081-layer1-rx3-conv3 | 0.4584 |
| 082-layer1-ry3-conv4 | 0.4940 |
| 083-layer1-rz3-conv5 | 0.3819 |
| 084-layer1-rzz23-conv6 | 0.4058 |
| 085-layer1-ryy23-conv7 | 0.4172 |
| 086-layer1-rxx23-conv8 | 0.4538 |
| 087-layer1-rx2-conv9 | 0.5512 |
| 088-layer1-ry2-conv10 | 0.4635 |
| 089-layer1-rz2-conv11 | 0.4970 |
| 090-layer1-rx3-conv12 | 0.8387 |
| 091-layer1-ry3-conv13 | 0.5864 |
| 092-layer1-rz3-conv14 | 0.7313 |
| 093-layer1-rx3-pool0 | 0.5705 |
| 094-layer1-ry3-pool1 | 0.6223 |
| 095-layer1-rz3-pool2 | 0.5464 |
| 096-layer1-rx2-pool3 | 0.4804 |
| 097-layer1-ry2-pool4 | 0.6717 |
| 098-layer1-rz2-pool5 | 0.0000 |
| 099-layer1-rz3--pool2 | 0.4451 |
| 100-layer1-ry3--pool1 | 0.4689 |
| 101-layer1-rx3--pool0 | 0.5451 |

## Codificación phase (40 entradas y pesos aleatorios)

- Características sin influencia en la salida: [] (cambio máximo por característica: [0.8401, 0.7306, 0.5579, 1.0835]).
- Pesos `[capa, k]` sin influencia: [[0, 'pool5'], [1, 'pool5']].
- Apariciones con contribución al gradiente: 99 de 102.
- Grado trigonométrico efectivo por peso `capa,k`: {'0,conv0': [4], '0,conv1': [4], '0,conv2': [4], '0,conv3': [4], '0,conv4': [4], '0,conv5': [4], '0,conv6': [4], '0,conv7': [4], '0,conv8': [4], '0,conv9': [4], '0,conv10': [4], '0,conv11': [4], '0,conv12': [4], '0,conv13': [4], '0,conv14': [4], '0,pool0': [4], '0,pool1': [4], '0,pool2': [4], '0,pool3': [2], '0,pool4': [2], '0,pool5': [0], '1,conv0': [1], '1,conv1': [1], '1,conv2': [1], '1,conv3': [1], '1,conv4': [1], '1,conv5': [1], '1,conv6': [1], '1,conv7': [1], '1,conv8': [1], '1,conv9': [1], '1,conv10': [1], '1,conv11': [1], '1,conv12': [1], '1,conv13': [1], '1,conv14': [1], '1,pool0': [2], '1,pool1': [2], '1,pool2': [2], '1,pool3': [1], '1,pool4': [1], '1,pool5': [0]}.
- Circuitos por muestra: parameter-shift 205 (actual) frente a 199 (solo apariciones vivas); secuencial 246 (actual) frente a 238 (grado efectivo).
- Diferencia máxima simulador rápido frente a Qibo: 6.7e-16.

| Aparición | Contribución máxima |
|---|---|
| 000-layer0-rx0-conv0 | 0.2731 |
| 001-layer0-ry0-conv1 | 0.4455 |
| 002-layer0-rz0-conv2 | 0.3874 |
| 003-layer0-rx1-conv3 | 0.3758 |
| 004-layer0-ry1-conv4 | 0.3754 |
| 005-layer0-rz1-conv5 | 0.3617 |
| 006-layer0-rzz01-conv6 | 0.4279 |
| 007-layer0-ryy01-conv7 | 0.4926 |
| 008-layer0-rxx01-conv8 | 0.5569 |
| 009-layer0-rx0-conv9 | 0.5225 |
| 010-layer0-ry0-conv10 | 0.4008 |
| 011-layer0-rz0-conv11 | 0.5536 |
| 012-layer0-rx1-conv12 | 0.5636 |
| 013-layer0-ry1-conv13 | 0.3781 |
| 014-layer0-rz1-conv14 | 0.4693 |
| 015-layer0-rx2-conv0 | 0.4099 |
| 016-layer0-ry2-conv1 | 0.4821 |
| 017-layer0-rz2-conv2 | 0.4104 |
| 018-layer0-rx3-conv3 | 0.3579 |
| 019-layer0-ry3-conv4 | 0.3033 |
| 020-layer0-rz3-conv5 | 0.4039 |
| 021-layer0-rzz23-conv6 | 0.4108 |
| 022-layer0-ryy23-conv7 | 0.3911 |
| 023-layer0-rxx23-conv8 | 0.3148 |
| 024-layer0-rx2-conv9 | 0.3142 |
| 025-layer0-ry2-conv10 | 0.2958 |
| 026-layer0-rz2-conv11 | 0.3238 |
| 027-layer0-rx3-conv12 | 0.4078 |
| 028-layer0-ry3-conv13 | 0.4747 |
| 029-layer0-rz3-conv14 | 0.3688 |
| 030-layer0-rx1-conv0 | 0.3269 |
| 031-layer0-ry1-conv1 | 0.4717 |
| 032-layer0-rz1-conv2 | 0.3188 |
| 033-layer0-rx2-conv3 | 0.3073 |
| 034-layer0-ry2-conv4 | 0.3923 |
| 035-layer0-rz2-conv5 | 0.2920 |
| 036-layer0-rzz12-conv6 | 0.3268 |
| 037-layer0-ryy12-conv7 | 0.3444 |
| 038-layer0-rxx12-conv8 | 0.3643 |
| 039-layer0-rx1-conv9 | 0.3874 |
| 040-layer0-ry1-conv10 | 0.4750 |
| 041-layer0-rz1-conv11 | 0.3666 |
| 042-layer0-rx2-conv12 | 0.4743 |
| 043-layer0-ry2-conv13 | 0.3697 |
| 044-layer0-rz2-conv14 | 0.4781 |
| 045-layer0-rx3-conv0 | 0.3643 |
| 046-layer0-ry3-conv1 | 0.4685 |
| 047-layer0-rz3-conv2 | 0.4090 |
| 048-layer0-rx0-conv3 | 0.2843 |
| 049-layer0-ry0-conv4 | 0.5034 |
| 050-layer0-rz0-conv5 | 0.3479 |
| 051-layer0-rzz30-conv6 | 0.5708 |
| 052-layer0-ryy30-conv7 | 0.3052 |
| 053-layer0-rxx30-conv8 | 0.4029 |
| 054-layer0-rx3-conv9 | 0.3792 |
| 055-layer0-ry3-conv10 | 0.3676 |
| 056-layer0-rz3-conv11 | 0.5168 |
| 057-layer0-rx0-conv12 | 0.3440 |
| 058-layer0-ry0-conv13 | 0.2722 |
| 059-layer0-rz0-conv14 | 0.3942 |
| 060-layer0-rx2-pool0 | 0.4965 |
| 061-layer0-ry2-pool1 | 0.3858 |
| 062-layer0-rz2-pool2 | 0.4825 |
| 063-layer0-rx0-pool3 | 0.2482 |
| 064-layer0-ry0-pool4 | 0.4115 |
| 065-layer0-rz0-pool5 | 0.0000 |
| 066-layer0-rz2--pool2 | 0.4874 |
| 067-layer0-ry2--pool1 | 0.3050 |
| 068-layer0-rx2--pool0 | 0.3997 |
| 069-layer0-rx3-pool0 | 0.4482 |
| 070-layer0-ry3-pool1 | 0.3717 |
| 071-layer0-rz3-pool2 | 0.3685 |
| 072-layer0-rx1-pool3 | 0.2868 |
| 073-layer0-ry1-pool4 | 0.5498 |
| 074-layer0-rz1-pool5 | 0.0000 |
| 075-layer0-rz3--pool2 | 0.3158 |
| 076-layer0-ry3--pool1 | 0.4581 |
| 077-layer0-rx3--pool0 | 0.2893 |
| 078-layer1-rx2-conv0 | 0.3997 |
| 079-layer1-ry2-conv1 | 0.4153 |
| 080-layer1-rz2-conv2 | 0.4075 |
| 081-layer1-rx3-conv3 | 0.2893 |
| 082-layer1-ry3-conv4 | 0.2674 |
| 083-layer1-rz3-conv5 | 0.4428 |
| 084-layer1-rzz23-conv6 | 0.4963 |
| 085-layer1-ryy23-conv7 | 0.3577 |
| 086-layer1-rxx23-conv8 | 0.3582 |
| 087-layer1-rx2-conv9 | 0.3627 |
| 088-layer1-ry2-conv10 | 0.4923 |
| 089-layer1-rz2-conv11 | 0.4444 |
| 090-layer1-rx3-conv12 | 0.4873 |
| 091-layer1-ry3-conv13 | 0.4615 |
| 092-layer1-rz3-conv14 | 0.2458 |
| 093-layer1-rx3-pool0 | 0.5835 |
| 094-layer1-ry3-pool1 | 0.3970 |
| 095-layer1-rz3-pool2 | 0.4428 |
| 096-layer1-rx2-pool3 | 0.5205 |
| 097-layer1-ry2-pool4 | 0.4793 |
| 098-layer1-rz2-pool5 | 0.0000 |
| 099-layer1-rz3--pool2 | 0.5906 |
| 100-layer1-ry3--pool1 | 0.3579 |
| 101-layer1-rx3--pool0 | 0.6102 |
