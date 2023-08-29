import { Range } from 'immutable';
import * as fs from 'fs';
import * as tf from '@tensorflow/tfjs-node';

function shuffle (array: number[]): void {
    for (let i = 0; i < array.length; i++) {
        const j = Math.floor(Math.random() * i)
        const swap = array[i]
        array[i] = array[j]
        array[j] = swap
    }
}

interface DataConfig {
    labels: number[],
    shuffle?: boolean,
    validationSplit?: number
}

export interface DataSplit {
    train: tf.data.Dataset<tf.TensorContainer>,
    validation?: tf.data.Dataset<tf.TensorContainer>
}

async function readImageFrom(imagePath: string): Promise<tf.Tensor3D> {
    return tf.node.decodeImage(fs.readFileSync(imagePath), 3) as tf.Tensor3D;
}

async function buildDataset(imagePaths: string[], labels: number[], indices: number[], config: DataConfig): Promise<tf.data.Dataset<tf.TensorContainer>> {
    // async function* dataGenerator(): AsyncGenerator<tf.TensorContainer> {
    //     for (let i = 0; i < indices.length; i++) {
    //         let image = await readImageFrom(imagePaths[indices[i]]);

    //         image = image.resizeBilinear([224, 224]).div(tf.scalar(255));
    //         // console.log(i + ": " + image.shape + ", " + imagePaths[indices[i]]);
           
    //         const label = labels[indices[i]];
    //         const sample = { xs: image, ys: [tf.scalar(label), tf.scalar(label)] };

    //         yield sample;
    //     }
    // }

    // // @ts-expect-error: For some reasons typescript refuses async generator but tensorflow do work with them
    // const dataset: tf.data.Dataset<tf.TensorContainer> = tf.data.generator(dataGenerator);

    async function* inputGenerator (): AsyncGenerator<tf.Tensor> {
        for (let i = 0; i < indices.length; i++) {
            let image = await readImageFrom(imagePaths[indices[i]]);
            image = image.resizeBilinear([224, 224]).div(tf.scalar(127.5)).sub(tf.scalar(1.0));
            yield image;
        }
    } 

    function* labelGenerator (): Generator<any> {
        for (let i = 0; i < indices.length; i++) {
            const label = labels[indices[i]];
            const labelOutput = { 
                'logits': tf.tensor(label), 
                'min_distances': tf.tensor(label)
            }
            yield labelOutput;
        }
    }

    //@ts-expect-error
    const xs: tf.data.Dataset<tf.TensorContainer> = tf.data.generator(inputGenerator);
    const ys: tf.data.Dataset<tf.TensorContainerArray> = tf.data.generator(labelGenerator);

    const dataset = tf.data.zip({ xs, ys });

    return dataset;
}

export async function loadAll(imagePaths: string[], config: DataConfig): Promise<DataSplit> {
    let labels: number[] = [];
    
    const indices = Range(0, imagePaths.length).toArray();
    const numberOfClasses = new Set(config.labels).size;
    labels = tf.oneHot(tf.tensor1d(config.labels, 'int32'), numberOfClasses).arraySync() as number[];

    if (config.shuffle === undefined || config.shuffle) {
        shuffle(indices);
    }
    
    if (config.validationSplit === undefined || config.validationSplit === 0) {
        const dataset = await buildDataset(imagePaths, labels, indices, config);
        return {
            train: dataset,
            validation: undefined
        }
    }

    const trainSize = Math.floor(imagePaths.length * (1 - config.validationSplit));

    const trainIndicies = indices.slice(0, trainSize);
    const validIndices = indices.slice(trainSize);

    const trainDataset = await buildDataset(imagePaths, labels, trainIndicies, config);
    const validDataset = await buildDataset(imagePaths, labels, validIndices, config);
    
    return {
        train: trainDataset,
        validation: validDataset
    }
}
