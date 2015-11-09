#Ejercicio 1

Resultados de Precision, Recall y F1 para modelos baseline:

- Flat
    - Parsed 1444 sentences
    - Labeled
        - Precision: 99.93%
        - Recall: 14.57%
        - F1: 25.43%

    - Unlabeled
        - Precision: 100.00%
        - Recall: 14.58%
        - F1: 25.45%
        Parsed 1444 sentences

- RBranch
    - Parsed Sents: 1444 sentences
    - Labeled
        - Precision: 8.81%
        - Recall: 14.57%
        - F1: 10.98%

    - Unlabeled
        - Precision: 8.87%
        - Recall: 14.68%
        - F1: 11.06%

-LBranch
    - Parsed Sents: 1444 sentences
    - Labeled
        - Precision: 8.81%
        - Recall: 14.57%
        - F1: 10.98%

    - Unlabeled
        - Precision: 14.71%
        - Recall: 24.33%
        - F1: 18.33%

# Ejercicio 2

Implementación de CKY parser.
Algoritmo dinámico para encontrar argmax de p(t) dada una oración tal que t es un parseo de la oración.

# Ejercicio 3

## Resultados

### UPCFG horzMarkov = None

+ 100.0% (1444/1444) (P=73.24%, R=72.94%, F1=73.09%)
+ Parsed 1444 sentences
+ Labeled
    - Precision: 73.24%
    - Recall: 72.94%
    - F1: 73.09%

+ Unlabeled
    - Precision: 75.35%
    - Recall: 75.04%
    - F1: 75.20%

- real    3m17.482s
- user    3m16.949s
- sys 0m0.312s

### UPCFG horzMarkov = 0

+ 100.0% (1444/1444) (P=70.25%, R=70.02%, F1=70.14%)
+ Parsed 1444 sentences
+ Labeled
    - Precision: 70.25%
    - Recall: 70.02%
    - F1: 70.14%

+ Unlabeled
    - Precision: 72.11%
    - Recall: 71.88%
    - F1: 72.00%

- real    1m15.997s
- user    1m15.947s
- sys 0m0.084s

### UPCFG horzMarkov = 1

+ 100.0% (1444/1444) (P=74.69%, R=74.60%, F1=74.64%)
+ Parsed 1444 sentences
+ Labeled
    - Precision: 74.69%
    - Recall: 74.60%
    - F1: 74.64%

+ Unlabeled
    - Precision: 76.56%
    - Recall: 76.46%
    - F1: 76.51%

- real    1m31.717s
- user    1m31.545s
- sys 0m0.120s

### UPCFG horzMarkov = 2

+ 100.0% (1444/1444) (P=74.78%, R=74.26%, F1=74.52%)
+ Parsed 1444 sentences
+ Labeled
    - Precision: 74.78%
    - Recall: 74.26%
    - F1: 74.52%

+ Unlabeled
 - Precision: 76.70%
 - Recall: 76.17%
 - F1: 76.43%

- real    2m21.653s
- user    2m21.596s
- sys 0m0.127s

### UPCFG horzMarkov = 3

+ 100.0% (1444/1444) (P=74.09%, R=73.46%, F1=73.77%)
+ Parsed 1444 sentences
+ Labeled
    - Precision: 74.09%
    - Recall: 73.46%
    - F1: 73.77%

+ Unlabeled
    - Precision: 76.25%
    - Recall: 75.60%
    - F1: 75.92%

- real    2m52.452s
- user    2m52.215s
- sys 0m0.288s

