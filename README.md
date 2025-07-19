# skin-detection

Wie Sie wissen, liegt der Farbbereich menschlicher Hauttypen im RGB-Farbbereich.
In diesem Projekt wird der Bayesian Classifier verwendet und auf unseren Daten werden Single-Gauß- und Multi-Gauß-Algorithmen implementiert.
Normalerweise berechnen wir p(skin|RGB), um mithilfe eines Schwellenwerts zu entscheiden, ob es sich bei dem Pixel um ein Skin-Pixel handelt oder nicht. Wir benötigen also nicht die genaue Wahrscheinlichkeit von p(skin|RGB) I, sondern können stattdessen eine Schätzung wie q(skin|RGB) haben, sodass, wenn p(skin|RGB)>Schwelle, dann q(skin |RGB)>Schwelle und Laster. Wenn wir eine solche Näherung berechnen können, werden die Ergebnisse ähnlich sein.


q(skin∣RGB)>Schwelle.


In diesem Projekt wird ein Bayesscher Klassifikator zur Hauterkennung im RGB-Farbraum verwendet. Dabei kommen sowohl der Single-Gauss- als auch der Multi-Gauss-Ansatz zur Anwendung.

Ziel ist es, auf Basis der RGB-Farbwerte eines Pixels die Wahrscheinlichkeit 
𝑝
(
skin
∣
RGB
)
p(skin∣RGB) zu bestimmen und mithilfe eines definierten Schwellenwerts zu entscheiden, ob es sich um ein Hautpixel handelt.

Anstelle der exakten Berechnung von 
𝑝
(
skin
∣
RGB
)
p(skin∣RGB) kann eine approximierte Funktion 
𝑞
(
skin
∣
RGB
)
q(skin∣RGB) verwendet werden, solange gewährleistet ist, dass bei 
𝑝
(
skin
∣
RGB
)
>
Schwelle
p(skin∣RGB)>Schwelle auch 
𝑞
(
skin
∣
RGB
)
>
Schwelle
q(skin∣RGB)>Schwelle gilt.

Diese Näherung erlaubt eine effizientere Verarbeitung, ohne die Klassifikationsergebnisse wesentlich zu beeinträchtigen.
