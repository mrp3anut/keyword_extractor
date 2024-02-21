# keyword_extractor
A python package to extract keywords using SpaCy and Sentence-Transformers.

# Installation
Clone/download the package files and

```
python setup.py install
```

You will also need to install a SpaCy model with

```
python -m spacy download en_core_web_sm
```

# Example
## Single Topic
An example list of texts about photolitography.

```
    texts = [
        "Photolithography is a process used in microfabrication to pattern parts of a thin film or the bulk of a substrate.",
        "It uses light to transfer a geometric pattern from a photomask to a light-sensitive chemical photoresist on the substrate.",
        "A series of chemical treatments then either etches the exposure pattern into the material or enables deposition of a new material in the desired pattern upon the material underneath the photoresist.",
        "The process begins with the substrate being coated with a light-sensitive photoresist.",
        "The coated substrate is then exposed to ultraviolet light through a photomask, which blocks light in some areas and allows it to pass in others, transferring the pattern.",
        "After exposure, the substrate is developed, washing away the photoresist in areas that were exposed to light, revealing the underlying substrate.",
        "If the process is negative, the photoresist remains on the substrate in areas that were exposed to light, protecting these areas in the subsequent etching process.",
        "The substrate may then undergo an etching process, which removes the exposed areas of the substrate not protected by photoresist.",
        "Alternatively, the developed substrate can be used in a deposition process, where new material is added to the areas not covered by photoresist.",
        "Finally, the remaining photoresist is removed, completing the patterning process."
    ]
```
Import the ```extract_keyword``` function, give a list of texts and arguments.

```
    from keyword_extractor import extract_keyword
    keyword = extract_keyword(texts, device='mps')
```
Output:
```
{'light - sensitive chemical photoresist': 0.85348475, 'light - sensitive photoresist': 0.84810925, 'remaining photoresist': 0.82319486, 'photoresist': 0.8205203, 'photolithography': 0.8144062}
```
## Multiple Topics

When you use multiple topics, the keywords are weighted by their cross-cluster term frequency.

```
texts_dict = {'photolithography': ["Photolithography is a process used in microfabrication to pattern parts of a thin film or the bulk of a substrate.",
        "It uses light to transfer a geometric pattern from a photomask to a light-sensitive chemical photoresist on the substrate.",
        "A series of chemical treatments then either etches the exposure pattern into the material or enables deposition of a new material in the desired pattern upon the material underneath the photoresist.",
        "The process begins with the substrate being coated with a light-sensitive photoresist.",
        "The coated substrate is then exposed to ultraviolet light through a photomask, which blocks light in some areas and allows it to pass in others, transferring the pattern.",
        "After exposure, the substrate is developed, washing away the photoresist in areas that were exposed to light, revealing the underlying substrate.",
        "If the process is negative, the photoresist remains on the substrate in areas that were exposed to light, protecting these areas in the subsequent etching process.",
        "The substrate may then undergo an etching process, which removes the exposed areas of the substrate not protected by photoresist.",
        "Alternatively, the developed substrate can be used in a deposition process, where new material is added to the areas not covered by photoresist.",
        "Finally, the remaining photoresist is removed, completing the patterning process."],

        'insulating film': ["A method of manufacturing a semiconductor device includes forming a first insulating film on a semiconductor substrate.",
        "A first opening is formed in the first insulating film.",
        "A first conductive film is formed on the first insulating film and in the first opening.",
        "A second insulating film is formed on the first conductive film.",
        "A second opening is formed in the second insulating film."]}
```
Import the ```extract_keyword``` function, give a dict of list of text where keys are the topic identifiers and values contain list of texts.
```
    from keyword_extractor import extract_keyword
    keyword = extract_keyword(texts_dict, device='mps')
```
Output:
```
{'photolithography': {'light - sensitive chemical photoresist': 1.1104092603237703, 'light - sensitive photoresist': 1.1034155677356225, 'remaining photoresist': 1.0710012069735289, 'photoresist': 1.0675214986616473, 'photolithography': 1.0595669158592786}, 'insulating film': {'second insulating film': 1.135674523446526, 'insulating film': 1.1080307255283068, 'conductive film': 1.0217540907066598, 'semiconductor substrate': 1.0198053237723224, 'semiconductor device': 1.0151911740962378}}
```
