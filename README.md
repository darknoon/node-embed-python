# Python Node Embed

The power of the existing Python ecosystem inside your Node.js or Electron environment.

Let's run PyTorch model on a high-end GPU, and display the results in WebGL:

```javascript
const python = require("node-embed-python");
const { importModule: py } = python;

// Define model in Python
const { run_my_model: runModel } = py`
import numpy as np
import torch

device = torch.device("cuda")

# Input is tensor of size []
def run_my_model(x):
    return biggan(x).cpu().numpy()
`;

// Integrate model with interactive javascript code
const np = python.import("numpy");

const x = np.randn([1, 2, 3]);

const result = runModel(x);
// Get typed array with raw data from python
const data = result.data();

const context = getCanvasGL();
drawBuffer(context, data);
```

# Installation

Get it from NPM:

```sh
$ yarn add node-embed-python
```

## Compatability

Use

- Node.js (N-API support required)
- Electron
- TODO: test in NW.js

## Requirements

The necessary node bindings will be built on installation. In the future, prebuilt binaries will be available.

- cmake >= 3.10
- ninja
- Python 3.x with Numpy

### Mac:

- Xcode developer tools (llvm)

### Windows:

Not yet supported

## Acknowledgements
