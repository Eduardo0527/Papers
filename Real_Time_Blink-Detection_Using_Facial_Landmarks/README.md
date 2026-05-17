# [Real-Time Eye Blink Detection using Facial Landmarks](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)

Using the pre-trained shape_predictor_68_face_landmarks, we extract the live feed from the users webcam and feed it into the model, once the landmark is drawn we extract the Eye aspect ratio from each eye: 
    $$\text{EAR}=\frac{\|p_2-p_6\|+\|p_3-p_5\|}{2\|p_1-p_4\|}$$
Then to we take the average of both EARs to calculate the final EAR.

For each frame a 13-dimensional feature vector is extracted by concatenating the EARs of its 6 neighboring frames. The authors opted to do that for each frame except the the beginning and ending of a video sequence. Since we are using live feed from the users webcam, a different approach was used. We used a sliding window buffer implemented using a fixed size deque from pythons collection library.

