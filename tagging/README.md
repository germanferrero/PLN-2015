#Descripción de tags.


Información básica:
sents: 17379
words(total): 517268
words(vocabulary): 46482
tags(vocabulary): 48


Información de tags:
Tags más frecue
ntes: ['nc', 'sp', 'da', 'vm', 'aq', 'fc', 'np', 'fp', 'rg', 'cc']

Significados:

* nc: Nombre Común
* sp: Preposición
* da: Artículo
* vm: Verbo principal
* aq: Adjetivo calificativo
* fc: Puntuación
* np: Nombre propio
* fp: Puntuación
* rg: Adverbio general
* cc: Conjunción coordinada

Tag: nc

* ocurrences: 92002
* percentaje: 17.78%
* Common words: ['años', 'presidente', 'millones', 'equipo', 'partido']

Tag: sp

* ocurrences: 79904
* percentaje: 15.44%
* Common words: ['de', 'en', 'a', 'del', 'con']

Tag: da

* ocurrences: 54552
* percentaje: 10.54%
* Common words: ['la', 'el', 'los', 'las', 'El']

Tag: vm

* ocurrences: 50609
* percentaje: 9.78%
* Common words: ['está', 'tiene', 'dijo', 'puede', 'hace']

Tag: aq

* ocurrences: 33904
* percentaje: 6.55%
* Common words: ['pasado', 'gran', 'mayor', 'nuevo', 'próximo']

Tag: fc

* ocurrences: 30148
* percentaje: 5.82%
* Common words: [',']

Tag: np

* ocurrences: 29113
* percentaje: 5.62%
* Common words: ['Gobierno', 'España', 'PP', 'Barcelona', 'Madrid']

Tag: fp

* ocurrences: 21157
* percentaje: 4.09%
* Common words: ['.', '(', ')']

Tag: rg

* ocurrences: 15333
* percentaje: 2.96%
* Common words: ['más', 'hoy', 'también', 'ayer', 'ya']

Tag: cc

* ocurrences: 15023
* percentaje: 2.90%
* Common words: ['y', 'pero', 'o', 'Pero', 'e']


#Ejercicio 2

Resultados Baseline:

- Accuracy: 89.03%,
- Accuracy in known words: 95.34%
- Accuracy in unknown words: 31.80%


#Ejercicio 4

Resultados:

## n=1

100.0%(89.01%,95.32%,31.80%)

- Accuracy: 89.01%
- Accuracy in known words: 95.32%
- Accuracy in unknown words: 31.80%

- real    0m12.658s
- user    0m12.588s
- sys     0m0.052s

## n=2

100.0%(92.72%,97.61%,48.42%)

- Accuracy: 92.72%
- Accuracy in known words: 97.61%
- Accuracy in unknown words: 48.42%

- real    0m22.109s
- user    0m22.027s
- sys     0m0.068s

## n=3

100.0%(93.17%,97.67%,52.31%)

- Accuracy: 93.17%
- Accuracy in known words: 97.67%
- Accuracy in unknown words: 52.31%

- real    1m10.965s
- user    1m10.891s
- sys     0m0.088s


## n=4

100.0%(93.14%,97.44%,54.14%)

- Accuracy: 93.14%
- Accuracy in known words: 97.44%
- Accuracy in unknown words: 54.14%

- real    6m13.231s
- user    6m12.601s
- sys     0m0.616s

#Ejercicio 7

Accuracy pattern is (total accuracy,known words accuracy, unknown words accuracy)

## n=1

- Logistic Regression: 100.0%(92.70%,95.28%,69.32%)

- Linear SVC: 100.0%(94.43%,97.04%,70.82%)
    - real    0m19.866s
    - user    0m19.765s
    - sys     0m0.072s

- MultinomialNB: 100.0%(82.18%,85.85%,48.89%)


## n=2

- Logistic Regression: 100.0%(91.99%,94.55%,68.75%)

- Linear SVC: 100.0%(94.29%,96.91%,70.57%)

- MultinombialNB: 100.0%(76.46%,80.41%,40.68%)
    - real    40m31.716s
    - user    39m40.548s
    - sys     0m0.979s


## n=3

- Logistic Regression: 100.0%(92.18%,94.71%,69.21%)

- Linear SVC: 100.0%(94.40%,96.94%,71.38%)

- MultinomialNB: 100.0%(71.47%,75.09%,38.59%)
    - real    40m20.792s
    - user    39m23.467s
    - sys     0m0.975s

## n=4

- Logistic Regression: 100.0%(92.23%,94.72%,69.60%)
    - real    0m24.303s
    - user    0m24.212s
    - sys     0m0.084s

- Linear SVC: 100%(94.46%,96.96%,71.81%)
    - real    0m24.174s
    - user    0m24.086s
    - sys     0m0.080s

- MultinomialNB: 100.0%(68.20%,71.31%,40.01%)
    - real    35m18.847s
    - user    34m16.171s
    - sys     0m0.919s
