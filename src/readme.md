This pipeline loads the video and the video contians the autonomous drive ros rviz video recording

on the left side of the video there is a rgb video from the front camera and ther other half of the scrren is the perception and planning interface.
1. planned path is projected on the hd map road.
2. on top of the planned path the predicted trajectory is also projectd as dots on it .
3. all the objects in the scene are detected with 3d bboxes.


now with all the informaiton using the gemma4 model it shoudl load all the frames and then process the image and anlyze and generate the speech only when there is some action done by the vehicle. like stopping turning slowing down due to the passengers on the road etcc... 

finish the python modules i have just added the skeletion use the langgraph 
