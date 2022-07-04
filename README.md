# vision_hub
Roger's vision model hub for research prototyping.

## Supported models

Interactive segmentation

## TODOs

- Add functionality to auto-download trained weights
- Avoid loading all models in initialization - load it on the fly like torch.hub
- Provide a coherent interface to support any type of input
- Make it a python package so that no PYTHONPATH needs to be specified
- Add requirements to support pip/conda to install dependencies
