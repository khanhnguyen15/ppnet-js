tensorflowjs_converter \
    --input_format=keras \
    --output_format=tfjs_layers_model   \
    ./pretrained/keras/efficient_net_b0.h5 \
    ./pretrained/js/efficient_net_b0

tensorflowjs_converter \
    --input_format=keras \
    --output_format=tfjs_layers_model   \
    ./pretrained/keras/resnet50v2.h5 \
    ./pretrained/js/resnet50v2

tensorflowjs_converter \
    --input_format=keras \
    --output_format=tfjs_layers_model   \
    ./pretrained/keras/resnet50.h5 \
    ./pretrained/js/resnet50