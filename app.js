const tf = require("@tensorflow/tfjs");
const wasm = require("@tensorflow/tfjs-backend-wasm");
const faceapi = require("@vladmandic/face-api/dist/face-api.node-wasm.js"); // use this when using face-api in dev mode

const canvas = require("canvas");
const extractFrames = require("ffmpeg-extract-frame");
const ffmpegPath = require("@ffmpeg-installer/ffmpeg").path;
const ffmpeg = require("fluent-ffmpeg");
ffmpeg.setFfmpegPath(ffmpegPath);

async function face() {
  try {
    // const video = "https://firebasestorage.googleapis.com/v0/b/spacev-9ca8d.appspot.com/o/models%2FfaceVideo.mp4?alt=media&token=04bf4184-e310-4dbc-9266-2598e3deba1a"
    const photo = "saf.jpg";
    await extractFrames({
      input: "./1.mp4",
      output: "./frame-%d.png",
      offsets: 3000,
      // timestamps: '50%',
    });

    const { Canvas, Image, ImageData } = canvas;
    // wasm.setWasmPaths(
    //   "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/"
    // );
    await tf.setBackend("wasm");
    await tf.ready();

    await faceapi.env.monkeyPatch({ Canvas, Image });
    const img = await canvas.loadImage(photo);
    const frame1 = await canvas.loadImage("./frame-1.png");
    // const frame2 = await canvas.loadImage("./frame-2.png");
    // const frame3 = await canvas.loadImage("./frame-3.png");

    await faceapi.nets.ssdMobilenetv1.loadFromDisk("./models");
    await faceapi.nets.faceLandmark68Net.loadFromDisk("./models");
    await faceapi.nets.faceRecognitionNet.loadFromDisk("./models");
    await faceapi.nets.faceExpressionNet.loadFromDisk("./models");

    // await faceapi.nets.withFaceLandmarks.loadFromDisk
    // console.log(
    //   `Face detection model loaded from path ${faceapi.nets.ssdMobilenetv1}`
    // );
    const reference = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();

    const result1 = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();

    const face = await faceapi
      .detectSingleFace(img).withFaceLandmarks().withFaceExpressions().withFaceDescriptor()
   
    let faceExpress  =getDominantEmotion(face.expressions)
    // console.log(faceExpress)

    if(faceExpress === "sad"){
      console.log("Depressed")
    }
    // console.log(face.expressions);

    // const result2 = await faceapi.detectSingleFace(frame2).withFaceLandmarks().withFaceDescriptor();
    // const result3 = await faceapi.detectSingleFace(frame3).withFaceLandmarks().withFaceDescriptor();

    // console.log(detections,detections2)

    // if (result1 && result2 && result3 && reference) {
    if (result1) {
      const faceMatcher1 = new faceapi.FaceMatcher(result1);
      // const faceMatcher2 = new faceapi.FaceMatcher(result2);
      // const faceMatcher3 = new faceapi.FaceMatcher(result3);

      if (reference) {
        const bestMatch1 = faceMatcher1.findBestMatch(reference.descriptor);
        // const bestMatch2 = faceMatcher2.findBestMatch(reference.descriptor);
        // const bestMatch3 = faceMatcher3.findBestMatch(reference.descriptor);
        // console.log(bestMatch1._label);
        // Trying to match atleast three frames with the given image to identify the person
        const identity1 = bestMatch1._label;
        // const identity2 = bestMatch2._label
        // const identity3 = bestMatch3._label

        // if (identity1 == "person 1" && identity2 == "person 1" && identity3 == "person 1") {
        if (identity1 == "person 1") {
          // res.send({
          //   success: true,
          //   text: "Match",
          // });

          // console.log("res");
        } else {
          // res.send({
          //   success: false,
          //   text: "Not a Match",
          // });

          // console.log("res");
        }
      }
    }
  } catch (err) {
    console.log(err);
  }
}

face();



function getDominantEmotion(expressionScores) {
  let maxScore = 0;
  let dominantEmotion = '';

  for (const [emotion, score] of Object.entries(expressionScores)) {
    if (score > maxScore) {
      maxScore = score;
      dominantEmotion = emotion;
    }
  }

  return dominantEmotion;
}