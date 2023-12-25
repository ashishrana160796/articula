mkdir pre-trained-models
python3 -m venv venv # create virtual environment
source venv/bin/activate # activate the virtual environment
echo "$PWD" > venv/lib/python3.10/site-packages/articula.pth # add project path to load project utility modules
pip install -r requirements.txt # installing packages specified in requirements file
