const tf = require("@tensorflow/tfjs");
const wasm = require("@tensorflow/tfjs-backend-wasm");
const faceapi = require("@vladmandic/face-api/dist/face-api.node-wasm.js"); // use this when using face-api in dev mode

const canvas = require("canvas");

async function face() {
  try {
    const { Canvas, Image, ImageData } = canvas;
    wasm.setWasmPaths(
      "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/"
    );
    await tf.setBackend("wasm");
    await tf.ready();

    await faceapi.env.monkeyPatch({ Canvas, Image });
    const img = await canvas.loadImage("./8.jpg");
    const img2 = await canvas.loadImage("./7.jpg");
    // let detections = await faceapi.detectAllFaces(canvas, this.getSSNMobileOptions())
    // await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, 'models'))
    await faceapi.nets.ssdMobilenetv1.loadFromDisk("./models");
    // await faceapi.nets.tinyFaceDetector.loadFromDisk('./model')
    // await faceapi.nets.faceLandMakr.loadFromDisk('./model')
    await faceapi.nets.faceLandmark68Net.loadFromDisk("./models");
    await faceapi.nets.faceRecognitionNet.loadFromDisk("./models");
    const reference = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();

      // console.log(reference)
    const result = await faceapi
      .detectSingleFace(img2)
      .withFaceLandmarks()
      .withFaceDescriptor();

    // console.log(detections,detections2)

    if (result) {
      const faceMatcher = new faceapi.FaceMatcher(result);
    //   drawLandmarks(videoEl, $("#overlay").get(0), [result], withBoxes);

      if (reference) {
        const bestMatch = faceMatcher.findBestMatch(reference.descriptor);
        console.log("its a  match :",bestMatch);
      }
    }
  } catch (err) {
    console.log(err);
  }
}

face();
