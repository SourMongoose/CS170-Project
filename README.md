## Instructions

1. Start with a fresh installation of Anaconda, version 2019.10. Ensure Python 3.7 is installed.
2. Clone this repo and cd into it.
3. Run `conda create -n car python=3.7`
4. Run `conda activate car`
5. Run `pip install -r requirements.txt`. Ensure you have sufficient disk space, a stable internet connection, and that all dependencies get installed correctly.
6. Ensure that inputs are placed in the `inputs/` directory, and have names ending in `_50_.in`, `_100.in`, and `_200.in`, as appropriate.
6. Run `python generate_outputs.py`. If any dependencies are missing, install them using `pip`.
7. Outputs will be created in the `outputs/` folder.
8. Run `python generate_outputs.py` many more times - the code will only output solutions that are better than the ones it sees in the output directory. We ran this for several iterations over the course of about 2 days on our personal computers. The algorithm is randomized, so it is unlikely you will get the same outputs.
If you are on Windows, parallelization may not work correctly. Set `parallelize = False` in `generate_outputs.py`.

## Computing resources used

Only our personal computers were used.

## External libraries used

tspy: https://pypi.org/project/tspy/

## Performance

As of the evening of 12/6, our outputs have a "Quality Across All Inputs" score of about 37.55.