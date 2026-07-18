--README--
These two programs are designed to work as a security measure for your laptop. 
Essentially, if someone other than yourself tries to access your laptop, the laptop will close itself out after 3 seconds or so. 
This project uses OpenCV, along with the use of libraries MediaPipe,face_recognition, and pickle to accomplish this. 

The way it works is that you run face_saver to save the image of YOUR face, so that when cv-lockout is run, it's constantly checking to see whether your face is on the screen or not.
To save your face, press spacebar. 
If your face stays in screen, nothing happens, and you can use the computer as normal. 
But if it falls out of the screen, or someone other than yourself puts their face in the camera, the computer returns to the lock screen after about 3 seconds.

To prevent false lockouts, make sure that your full face is up close and visible to the camera, as the detection strength deteriorates with distance.

This is the initial version, hopefully greater functionality can be added in order to improve it.