import Rand from 'rand-seed';

import * as tf from '@tensorflow/tfjs-node';
import { DataSplit } from './javascripts/data_prep/image_loader';
import { loadData, loadTrainAndVal } from './javascripts/data_prep/birds_data';
import { logitLoss, protoPartLoss } from './javascripts/models/loss';
import { PPNet, convFeatures, getProtoClassIdx } from './javascripts/models/ppnet';

async function getData(): Promise<DataSplit> {
    // const dir = './data/CUB_200_2011/images/';
    // const dataSplit = await loadData(dir);    
    const trainDir = './data/cub200_cropped/train_cropped_augmented/';
    const valDir = './data/cub200_cropped/test_cropped/'; 
    const dataSplit = loadTrainAndVal(trainDir, valDir);

    return dataSplit;
}

async function getModel(): Promise<tf.LayersModel> {
    return PPNet({ backbone: 'resnet50', numClasses: 20 });
}

async function main(): Promise<void> {
    const model = await getModel();
    const dataset = await getData();

    const trainDataset = dataset['train'].batch(10);
    const validDataset = dataset['validation']?.batch(10);

    const protoClassId = await getProtoClassIdx({
        prototypeShape: [200, 1, 1, 128],
        numClasses: 20
    });

    const ppLoss = protoPartLoss(
        {
            prototypeShape: [200, 1, 1, 128]
        },
        protoClassId
    );

    model.compile(
        {
            optimizer: 'adam',
            loss: ['categoricalCrossentropy'],
            // loss: logitLoss,
            metrics: ['accuracy']
        }
    )

    await model.fitDataset(
        trainDataset, 
        {
            validationData: validDataset,
            epochs: 10
        }
    )

    console.log(); 
}

main().then(() => {
    console.log("Done");
});