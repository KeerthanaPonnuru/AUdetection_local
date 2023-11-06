# AU detection

For the video capture OpenCV has been utilized and for the web interface, Flask application has been used. 
## Installation
It is expected that Python (version 3 preferred) is installed.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Flask and Flask-CORS using pip. Make sure to install OpenCV.



```bash
pip install Flask

pip install flask-cors

pip install opencv-python

pip install opencv-python mediapipe
```

## Instructions
To run the code follow the steps mentioned below:

1. Open terminal/cmd and make sure to be in the same path as the code files
2. Enter the command 'python app.py'
3. Once the server is running, there would be an output message that includes the Flask application's URL like 'Running on http://127.0.0.1:5000' click on the URL
4. This would redirect you to the browser with the live video stream
5. If you would need to run the code again, enter this command 'lsof -i :5000' this returns a list of processes and associated network connections currently using port 5000 on your system
6. Next run the command 'sudo kill -9 pid' here pid is the PID of the earliest Python file listed by the previous command

```bash
lsof -i :5000

sudo kill -9 pid
```


