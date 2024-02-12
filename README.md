# keyword_extractor
A python package to extract keywords

# Installation
Clone/download the package files and

```
python setup.py install
```

You will also need to install a SpaCy model

```
python -m spacy download en_core_web_sm
```

# Example
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
{'photoresist': 0.8205202, 'remaining photoresist': 0.82319486, 'light - sensitive photoresist': 0.84810936, 'light - sensitive chemical photoresist': 0.85348475}
```
