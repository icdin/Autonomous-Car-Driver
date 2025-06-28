# Autonomous-Car-Driver
The project highlights how to use a simulator(Udacity) to program and run an autonomous vehicle
Requirements Install the following tools:
   1. python
   2. anaconda 
   
**Steps to run this simulator**
1. Clone this repo git clone https://github.com/icdin/autonomous-car-driver
2. Install Udacity https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip
   Extract the zip file and run the executable file .exec
3. Create your virtual environment run conda create -n myenv
4. Activate the environment run conda activate myenv
5. Install the following dependencies opencv-contrib-python, numpy matplotlib, tensorflow, Flask, eventlet, python-engineio, python-socketio Or for short,
   run pip -r requirements.txt to automatically install all.
6. Drive the autonomous car
   To drive this model, open a terminal from this environment and run "python drive.py --user" Once connected,
   go to the simulator and select a track and hit "Autonomous Mode" If everything went well, then the autonomous car will be seen driving itself.

**Disclaimer** 
This project is for educational purposes and shouldn't be installed on any machine for live usage. If the needs for a real live autonomous car driving arises, I would use Carla instead of Udacity. While Carla would require an Nvidia PC, Udacity runs smoothly on low graphic machines without crashing and this is the reason it is being used to prepare this project.
