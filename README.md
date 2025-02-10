# Overview

Real-time people detection with Slack integration to notify Trend Ottawa employees how long the line is to get food on Wednesdays.

Utilizes computer vision, image processing, and existing deep learning models to detect people in a live camera feed.

*Developed by Hadi Srour, Hajar Assim, Kien Do, and Jeyason Jeyaparan.*

## Setup
- Open the terminal in the root of the project
- Create a virtual environment
```console
python3 -m venv venv          # create the venv
source venv/bin/activate      # activate the venv
```
- Install project dependencies
```console
pip3 install opencv-python
pip3 install tensorflow
pip3 install slack_bolt
pip3 install slack_sdk
pip3 install slackclient
```
- Open a new terminal and run the people detection program
```console
python3 main.py
```
> [!NOTE]  
> Click "Allow" when your code editor asks for permission to open the camera.

- Open another terminal to start up the Slack server to send/receive messages
```console
python3 slack.py
```
- Use the bot by going to your channel's Integrations tab --> App --> Add "LunchBox AI"
- Begin using it with the following commands
```console
@LunchBoxAI
@LunchBoxAI /setmenu Chicken and rice
@LunchBoxAI /setmenu leftovers
@LunchBoxAI /setmenu end
```
