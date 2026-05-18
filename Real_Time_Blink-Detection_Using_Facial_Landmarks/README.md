# [Real-Time Eye Blink Detection using Facial Landmarks](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)

### Methodology 

**1. Facial Landmark Extraction** 
We capture a live video feed from the user's webcam and process each frame using the pre-trained `shape_predictor_68_face_landmarks` model.

**2. Eye Aspect Ratio (EAR) Calculation**
Once the facial landmarks are mapped, we extract the coordinates for both eyes and calculate the Eye Aspect Ratio (EAR) using the following equation:

$$\text{EAR}=\frac{\|p_2-p_6\|+\|p_3-p_5\|}{2\|p_1-p_4\|}$$

We then take the average of the left and right EARs to establish a single, stable EAR metric for the current frame.
<img width="223" height="147" alt="Captura de tela 2026-05-18 151342" src="https://github.com/user-attachments/assets/9d11a30e-70c1-483c-8733-49e4d78fbfe8" />

**3. Temporal Feature Vector**
A single frame's EAR is often insufficient to classify a blink accurately. To capture the temporal pattern of a blink (the closing and opening of the eye over time), we rely on a 13-dimensional feature vector. 

While the original authors generated this vector by concatenating the EARs of a frame and its 6 neighboring frames (forward and backward) from a pre-recorded video sequence, our real-time implementation requires a different approach. We utilize a sliding window buffer, implemented using a fixed-size `deque` from Python's `collections` library, to maintain a rolling history of the most recent frames.

**4 Classification and State Tracking**
Once the sliding window buffer contains 13 frames of EAR data, the vector is passed to a scikit-learn Linear Support Vector Machine (SVM) for binary classification. To help the linear model understand the temporal behavior of the sequence, we perform real-time feature engineering by calculating the mean, variance, and range of the 13 frames. The augmented feature vector is then normalized using a pre-fitted StandardScaler to optimize the distance-based SVM predictions.

Because a standard human blink spans multiple frames on a standard 30 FPS webcam, evaluating every frame independently would cause the system to count a single blink multiple times. To solve this, we implemented a debounce state-tracker. The system triggers a blink count only on the initial transition from "Awake" to "Blink," locking the state and ignoring subsequent "Blink" predictions until the user's eye registers as fully open again.
