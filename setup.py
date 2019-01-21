import os

start_dir = os.getcwd()

# tensorflow
os.system("sudo apt-get -y install python-pip python-dev")
os.system("sudo pip install tensorflow-gpu")

# tflearn
os.system("sudo pip install tflearn")
os.system("sudo apt-get -y install python-h5py")
os.system("sudo apt-get -y install python-scipy")

# matplotlib
os.system("sudo apt-get -y install python-matplotlib")


