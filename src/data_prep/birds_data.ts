import fs from 'fs';
import Rand from 'rand-seed';

import * as tf from '@tensorflow/tfjs-node';
import { DataSplit, loadAll } from './image_loader';

const rand = new Rand('1234');

function shuffle<T, U> (array: T[], arrayTwo: U[]): void {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(rand.next() * (i + 1))
        const temp = array[i]
        array[i] = array[j]
        array[j] = temp
    
        const tempTwo = arrayTwo[i]
        arrayTwo[i] = arrayTwo[j]
        arrayTwo[j] = tempTwo
    }
}

function subfoldersFromRoot(dir: string): string[] {
    const f = fs.readdirSync(dir, { withFileTypes: true });
    return f.filter(file => file.isDirectory())
            .map(folder => dir + folder.name);
}

function filesFromFolder(folder: string): string[] {
    const f = fs.readdirSync(folder);
    return f.map(file => folder + '/' + file);
}

export async function loadData(dir: string): Promise<DataSplit> {
    const classFolders: string[] = subfoldersFromRoot(dir);
    const classFiles: string[][] = classFolders.map(folder => filesFromFolder(folder)).slice(0, 20);

    const labels = classFiles.flatMap((files, index) => Array(files.length).fill(index));
    const files = classFiles.flat(); 

    shuffle(files, labels);

    return await loadAll(files, { labels: labels, shuffle: true, validationSplit: 0.2 });
}

export async function loadTrainAndVal(trainDir: string, valDir: string): Promise<DataSplit> {
    const trainFolders: string[] = subfoldersFromRoot(trainDir);
    const trainFiles: string[][] = trainFolders.map(folder => filesFromFolder(folder));
    const valFolders: string[] = subfoldersFromRoot(valDir);
    const valFiles: string[][] = valFolders.map(folder => filesFromFolder(folder));

    const trainLabels = trainFiles.flatMap((files, index) => Array(files.length).fill(index));
    const trainFlatFiles = trainFiles.flat(); 
    const valLabels = valFiles.flatMap((files, index) => Array(files.length).fill(index));
    const valFlatFiles = valFiles.flat(); 

    shuffle(trainLabels, trainFlatFiles);
    shuffle(valLabels, valFlatFiles);
    
    const trainData = await loadAll(trainFlatFiles, { labels: trainLabels, shuffle: true, validationSplit: 0.0 });
    const valData = await loadAll(valFlatFiles, { labels: valLabels, shuffle: false, validationSplit:1.0 });

    return {
        train: trainData.train,
        validation: valData.validation
    }    
}