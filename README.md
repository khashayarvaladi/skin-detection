# skin-detection



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


This project uses a Bayesian classifier to detect human skin pixels in the RGB color space. Both single-Gaussian and multi-Gaussian approaches are implemented.

The goal is to estimate the probability 
𝑝
(
skin
∣
RGB
)
p(skin∣RGB) based on a pixel's RGB values, and to decide whether it represents skin using a defined threshold.

Instead of computing the exact probability, an approximation 
𝑞
(
skin
∣
RGB
)
q(skin∣RGB) can be used, as long as the condition 
𝑝
(
skin
∣
RGB
)
>
threshold
⇒
𝑞
(
skin
∣
RGB
)
>
threshold
p(skin∣RGB)>threshold⇒q(skin∣RGB)>threshold holds.

This allows for more efficient processing while maintaining comparable classification results.
