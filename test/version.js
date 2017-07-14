'use strict'
const tensorflow = require('bindings')('tensorflow')
console.log(tensorflow.sessionRun())
