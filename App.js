import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { LogBox, StyleSheet, View} from 'react-native';

import * as tf from "@tensorflow/tfjs"

import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection"

import { Camera } from 'expo-camera';
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';

const TensorCamera = cameraWithTensors(Camera);
let model = null;
LogBox.ignoreAllLogs(true)


function euclidean_dist (x1, y1, x2, y2) {
  return Math.sqrt( Math.pow((x1-x2), 2) + Math.pow((y1-y2), 2) );
};

  

export default class App extends React.Component {

  constructor(props) {
    super(props)
    this.state= {
      tfready:false,
      cameraType: Camera.Constants.Type.front,
      hasPermission:null,
      faceDetector:null,
      enableDetections: true
    }
    this.handleCameraStream = this.handleCameraStream.bind(this);
  }

  async componentDidMount() {
    const { status } = await Camera.requestPermissionsAsync();

    this.setState({hasPermission: status==='granted'})
    
    await tf.ready();
    // await tf.setBackend("rn-webgl");
    
    model = await faceLandmarksDetection.load(
      faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
      {shouldLoadIrisModel:true});
      
    this.setState({faceDetector: model})
    this.setState({tfready:true});
  }

  componentWillUnmount() {
    if(this.rafID) {
      cancelAnimationFrame(this.rafID);
    }
  }

  async handleCameraStream(images, updatePreview, gl) {
    const loop = async () => {
      const nextImageTensor = images.next().value;
      if(this.state.faceDetector!=null && this.state.enableDetections===true) {
        const preds = await this.state.faceDetector.estimateFaces({input:nextImageTensor,returnTensors:false})
        preds.forEach((face) => {
          const leftEyeLower = face["annotations"]["leftEyeLower0"]
          const leftEyeUpper = face["annotations"]["leftEyeUpper0"]
          
          const leftCenterLower = leftEyeLower[4]
          const leftCenterUpper = leftEyeUpper[4]
          const leftLeft = leftEyeLower[0]
          const leftRight = leftEyeLower[8]

          const leftVertDist = euclidean_dist(leftCenterLower[0],leftCenterLower[1],leftCenterUpper[0],leftCenterUpper[1])
          const leftHorizDist = euclidean_dist(leftLeft[0],leftLeft[1],leftRight[0],leftRight[1])
          const leftClosedScore = leftVertDist / (2.*leftHorizDist)
          const leftClosed = leftClosedScore < .11 ? true : false;
         
          
          const rightEyeLower = face["annotations"]["rightEyeLower0"]
          const rightEyeUpper = face["annotations"]["rightEyeUpper0"]
          
          const rightCenterLower = rightEyeLower[4]
          const rightCenterUpper = rightEyeUpper[4]
          const rightLeft = rightEyeLower[0]
          const rightRight = rightEyeLower[8]

          const rightVertDist = euclidean_dist(rightCenterLower[0],rightCenterLower[1],rightCenterUpper[0],rightCenterUpper[1])
          const rightHorizDist = euclidean_dist(rightLeft[0],rightLeft[1],rightRight[0],rightRight[1])
          const rightClosedScore = rightVertDist / (2.*rightHorizDist)
          const rightClosed = rightClosedScore < .11 ? true : false;    

          console.log(leftClosedScore, rightClosedScore)

        })    
      } 
      tf.dispose(nextImageTensor)     
      this.rafID = requestAnimationFrame(loop);
    }
    loop();
  }


  render() {
    // Currently expo does not support automatically determining the
    // resolution of the camera texture used. So it must be determined
    // empirically for the supported devices and preview size. 
    let textureDims;
    if (Platform.OS === 'ios') {
     textureDims = {
       height: 1920,
       width: 1080,
     };
    } else {
     textureDims = {
       height: 1200,
       width: 1600,
     };
    }
 
    return (
    <View
    style={styles.container}>
      {this.state.tfready && (
        <TensorCamera
        // Standard Camera props
        style={styles.camera}
        type={this.state.cameraType}
        // Tensor related props
        cameraTextureHeight={textureDims.height}
        cameraTextureWidth={textureDims.width}
        resizeHeight={640}
        resizeWidth={480}
        resizeDepth={3}
        onReady={this.handleCameraStream}
        autorender={true}
       />
      )}
    </View>)
   }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    alignItems: 'center',
    justifyContent: 'center',
  },
  camera: {
    flex:1,
    width:'100%',
    height:"100%"
  }
});
