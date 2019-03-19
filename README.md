# Human Speech Processing
Decoding auditory representation of the brain using natural speech stimuli

## Data
1. [Bang! You're Dead](https://caltech.app.box.com/s/m4rztkl85uavl2xuv5eir8y0ysjn657c)
- [Script](https://caltech.app.box.com/s/jq2oclsr8jy6lie11ghkgoko7pcu3occ)
- [Audio](https://caltech.app.box.com/s/knzckzcfiq1s3anm21166bf17tlsnjtr)
- about 30-40 normal subjects scanned at Caltech 
- about 15 epileptic patients scanned at Caltech + recorded intracranially in the hospital(sEEG: patients except 6 and 8)
- about 700 subjects scanned in Cambridge

2. [Forest Gump](http://studyforrest.org)
- [Script](https://github.com/psychoinformatics-de/studyforrest-data-annotations)
- fMRI
- sEEG (patient number 6 and 8)

## Candidate features
|features|regions|refrences|notes|
|---|---|---|---|
|Word level semantics|MTG, MFG, IFG|[de Heer 2017](http://www.jneurosci.org/content/37/27/6539)||
|Sentence level semantics|MTG, MFG, IFG|[Fedorenko 2011](https://www.pnas.org/content/108/39/16428), [Huth 2016](https://www.nature.com/articles/nature17637)||
|Syntax and discourse, especially about identities of characters in a story|MTG, IFG|[Wehbe 2014](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0112575)||
|Onset of sentences|MTG|[Hamilton 2018](https://www.cell.com/current-biology/fulltext/S0960-9822(18)30461-5)||
|Intonation|MTG, MFG, IFG|[Tang 2017](http://science.sciencemag.org/content/357/6353/797)|Need synthesized stimuli|
|Coherence of a word, sentence & paragraph order|MTG, MFG, IFG|[Lerner 2011](https://www.ncbi.nlm.nih.gov/pubmed/21414912)|Need synthesized stimuli|

## Tools
1. [pliers(feature extration)](https://github.com/tyarkoni/pliers#user-guide)
2. [The Penn Phonetics Lab Forced Aligner](https://babel.ling.upenn.edu/phonetics/old_website_2015/p2fa/index.html)
3. [DA tagger](https://github.com/ColingPaper2018/DialogueAct-Tagger)
4. [STT](https://github.com/GoogleCloudPlatform/python-docs-samples/tree/master/speech/cloud-client)

## Dependencies
- Python3.7
- nibabel, nilearn, ninype
- textblob, spacy
- [fmriprep](https://github.com/poldracklab/fmriprep)
- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL)
- [DA tagger](https://github.com/ColingPaper2018/DialogueAct-Tagger)

## Usage
To train DA tagger
```bash
python DialogueAct-Tagger/scripts/train.py
```
To transcribe the audio file
```bash
python src/transcribe.py <path_to_the_audio_file>
```
To extract features, load the brain data and fit the encoding models
```bash
python src/main.py 
```

