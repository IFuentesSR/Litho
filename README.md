# Analysis and Clustering of Lithological Descriptions


## Development setup

Use (ana)conda environments to keep everything nice, consistent, and cleanly separated.

The provided environment.yml file can be used to setup the necessary conda environment.

In the project directory via the terminal/command line:

`conda create env`

This will set up the development environment (after saying 'yes' to installation), which can be activated with:

`activate lith_nlp` on Windows

and

`source activate lith_nlp` on *nix-like systems

### Example setup of Jupyter kernel

`python -m ipykernel install --user --name lith-nlp --display-name "Lith NLP"`

### Other setup information

This project requires the following nltk data to be downloaded

```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

Add the target directory path as the second parameter if necessary:

```
import nltk
nltk.download('stopwords', 'C:/a_directory/path')
```
