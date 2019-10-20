
# Hand/Gesture Detector

Started this project with the intent to feed finger tip coordinates into some kind of classifier
(bayes, NN, RNN for video) to detect gestures without running an entire CNN on an image, or maybe doing an ensemble .

## Improvements I want to make
- face removal (figure out WHICH contour is the most hand-like one) 
- ideas to automatically find skin color even in somewhat dynamic lighting
    - continuously adjust baseline color to median or mean of matching region
    - sample from neighboring regions and "flood-fill" towards lowest different (when region is unrealistically small)
    - eliminate some convexity defects (where the defect is extreme like in the middle of the palm)
    - train and use an SVM or lightweight classifier to find it
- maybe just use the mask and maybe contours to generate image to use in CNN? 
- ensemble that with predictions based on finger coordinates/distances? 
- smooth jitter by buffering frames and:
  - ignoring/removing frames that have an outlier for hand_center/radius
  - taking mean coordinates to smooth motion
- use this to generate training data for CNN or maybe some kind of GANs thing 


## Potential applications
- ASL transcription via NN or other classifier
- Video Game controls
- TV remote
- AR/VR desktop control (add eye tracking and stuff)
- lightweight general purpose gesture detector API
