export function computeLayerRfInfo(
    layerFilterSize: number,
    layerStride: number,
    layerPadding: 'same' | 'valid' | number,
    previousLayerRfInfo: [number, number, number, number]
): [number, number, number, number] {
    const [nIn, jIn, rIn, startIn] = previousLayerRfInfo;

    let nOut, pad;
    if (layerPadding === 'same') {
        nOut = Math.ceil(nIn / layerStride);
        if (nIn % layerStride === 0) {
            pad = Math.max(layerFilterSize - layerStride, 0);
        } else {
            pad = Math.max(layerFilterSize - (nIn % layerStride), 0);
        }
    } else if (layerPadding === 'valid') {
        nOut = Math.ceil((nIn - layerFilterSize + 1) / layerStride);
        pad = 0;
    } else {
        pad = (layerPadding as number) * 2;
        nOut = Math.floor((nIn - layerFilterSize + pad) / layerStride) + 1;
    }

    const pL = Math.floor(pad / 2);

    const jOut = jIn * layerStride;
    const rOut = rIn + (layerFilterSize - 1) * jIn;
    const startOut = startIn + ((layerFilterSize - 1) / 2 - pL) * jIn;

    return [nOut, jOut, rOut, startOut];
}

export function computeProtoLayerRfInfo(
    img_size: number,
    layerFilterSizes: number[],
    layerStrides: number[],
    layerPaddings: ('same' | 'valid' | number)[],
    prototypeKernelSize: number
): [number, number, number, number] {
    console.assert(layerFilterSizes.length === layerStrides.length);
    console.assert(layerFilterSizes.length === layerPaddings.length);

    let rfInfo: [number, number, number, number] = [img_size, 1, 1, 0.5];

    for (let i = 0; i < layerFilterSizes.length; i++) {
        const filterSize = layerFilterSizes[i];
        const strideSize = layerStrides[i];
        const paddingSize = layerPaddings[i];

        rfInfo = computeLayerRfInfo(
            filterSize,
            strideSize,
            paddingSize,
            rfInfo
        );
    }

    const protoLayerRfInfo = computeLayerRfInfo(
        prototypeKernelSize,
        1,
        'valid',
        rfInfo
    );

    return protoLayerRfInfo;
}
