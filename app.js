const tf = require("@tensorflow/tfjs");
const wasm = require("@tensorflow/tfjs-backend-wasm");
const faceapi = require("@vladmandic/face-api/dist/face-api.node-wasm.js"); // use this when using face-api in dev mode

const canvas = require("canvas");
const extractFrames = require('ffmpeg-extract-frame')
const ffmpegPath = require('@ffmpeg-installer/ffmpeg').path;
const ffmpeg = require('fluent-ffmpeg');
ffmpeg.setFfmpegPath(ffmpegPath);

async function face() {
  try {
    const { Canvas, Image, ImageData } = canvas;
    wasm.setWasmPaths(
      "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/"
    );
    await tf.setBackend("wasm");
    await tf.ready();

    await faceapi.env.monkeyPatch({ Canvas, Image });
    const img = await canvas.loadImage("./mukesh_1.jpg");
    const img2 = await canvas.loadImage("./apurv_2.jpg");
    // let detections = await faceapi.detectAllFaces(canvas, this.getSSNMobileOptions())
    // await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, 'models'))
    await faceapi.nets.ssdMobilenetv1.loadFromDisk("./models");
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
        console.log("Match Status :",bestMatch);
      }
    }
  } catch (err) {
    console.log(err);
  }
}

// face();



async function vidFace() {

  try{
    const { Canvas, Image, ImageData } = canvas;
    await faceapi.env.monkeyPatch({ Canvas, Image });
  // wasm.setWasmPaths(
  //   "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/"
  // );
  await tf.setBackend("wasm");
  await tf.ready();
  
  // let video = ('./1.mp4')
  let frames = [];
  let fps = 1; // Frames per seconds to
  let interval = 1 / fps; // Frame interval
  let maxDuration = 10; // 10 seconds max duration
  let currentTime = 0; //
  await faceapi.nets.ssdMobilenetv1.loadFromDisk("./models");
  await faceapi.nets.faceLandmark68Net.loadFromDisk("./models");
  await faceapi.nets.faceExpressionNet.loadFromDisk("./models");

  await extractFrames({
    input: '1.mp4',
    output: './frame-%d.png',
    offset: 2000
  })

  const frame = await canvas.loadImage('./frame-1.png')

  const dimensions = {
    width: frame.width,
    height: frame.height
};

  if(frame){

  const detectionWithExpressions = await faceapi.detectSingleFace(frame).withFaceLandmarks().withFaceExpressions()
 const reaction =  faceapi.resizeResults(detectionWithExpressions,dimensions)
    // console.log(reaction.expressions)

    if(reaction.expressions){

      let react = reaction.expressions

    //   for (let key in react) { // This will throw an error
    //     // prop = 'x
    //     console.log(key)
    //     console.lll
    // }
    let curReaction  ;
    for (let [key, value] of Object.entries(react)) {
      // console.log(`${key} value=${value}`)
      if(value > 0.1 && value < 1.0  ){
        curReaction = key.toUpperCase()
      }
  }
  console.log(curReaction)



    }
    
  }

// console.log(detectionWithExpressions)

  }
  catch(err){
console.log(err)
  }

}

vidFace()

