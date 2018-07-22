
import * as tf from '@tensorflow/tfjs';
import {MnistData} from './data';

function setupModel(filterSize){
  let model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 5,
    filters: filterSize,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  const LEARNING_RATE = 0.15;
  const optimizer = tf.train.sgd(LEARNING_RATE);
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}


async function showPredictions(filterSize) {
  const batch = data.nextTestBatch(1);
  let model = setupModel(filterSize);
  tf.tidy(() => {

    const input = batch.xs.reshape([-1, 28, 28, 1]);
    const output = model.predict(input);

    const inputData = input.dataSync();
    const outputData = output.dataSync();
    console.log("filterSize",filterSize);
    // console.log("input",inputData);
    // console.log("output",outputData);


    //fixed input size 28x28, no padding, strides[1,1], kernel size 5
    //Going to show first activation with inputs and weights and ouput
    for(let y = 0; y < 24; y++){
      for(let x = 0; x < 24; x++){
        let index = y * 24 + x;

        if(outputData[index] > 0){
          console.log(`First activation at [${x},${y}]`)

          let firstFilterWight = model.layers[0].getWeights()[0].dataSync().slice(0,25);
          console.log("weights of first filter",firstFilterWight);
      
          let inputWindow = []
          for(let yKernel = 0; yKernel < 5; yKernel++ ){
            let inputIndex = y * 28 + yKernel * 28 + x ;
            inputWindow.push(...inputData.slice(inputIndex,inputIndex+5));
          }
          console.log("Input window values",inputWindow);

          let dotProduct = 0;
          for(let i in inputWindow){
            dotProduct += inputWindow[i] * firstFilterWight[i];
          }


          //bias
          let bias = model.layers[0].getWeights()[1].dataSync()[0];
          dotProduct += bias;

          console.log("dot product",dotProduct);
          console.log(`First activation at [${x},${y}]`,outputData[index])

          return;
        }


      }
    }


  });
}

let data;
async function load() {
  data = new MnistData();
  await data.load();
}

async function mnist() {
  await load();
  showPredictions(1);
  showPredictions(2);
}

mnist();
