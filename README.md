## Prototypical Part Network in Javascript

This repo is an attempt to simplify and translate the archtecture of [Prototypical Part Network](https://github.com/cfchen-duke/ProtoPNet) into [Tensorflow.js](https://www.tensorflow.org/js). 

The translation is done from Pytorch -> Tensorflow -> Tensorflow.js using Typescript. A large proportion of the network has been simplified to have a minimum viable prototype.

The architecture is further integrated into [disco](https://github.com/epfml/disco/tree/wip-ppnet).

The main structure of the repo is:
```
root
  |___ data 
  |___ javascripts      # typescript tensorflow.js version
  |___ pretrained 
  |___ python
    |___ ppnet_tf       # tensorflow simplified version
    |___ ppnet          # pytorch version
  |___ test_model.ts    # testing scripts
  
```