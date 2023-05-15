# skin-detection

Wie Sie wissen, liegt der Farbbereich menschlicher Hauttypen im RGB-Farbbereich.
In diesem Projekt wird der Bayesian Classifier verwendet und auf unseren Daten werden Single-Gauß- und Multi-Gauß-Algorithmen implementiert.
Normalerweise berechnen wir p(skin|RGB), um mithilfe eines Schwellenwerts zu entscheiden, ob es sich bei dem Pixel um ein Skin-Pixel handelt oder nicht. Wir benötigen also nicht die genaue Wahrscheinlichkeit von p(skin|RGB) I, sondern können stattdessen eine Schätzung wie q(skin|RGB) haben, sodass, wenn p(skin|RGB)>Schwelle, dann q(skin |RGB)>Schwelle und Laster. Wenn wir eine solche Näherung berechnen können, werden die Ergebnisse ähnlich sein.
