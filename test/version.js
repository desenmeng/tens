'use strict'
const tensorflow = require('bindings')('tensorflow')
const proto = require('protobufjs');
const str = tensorflow.version();
console.log(str);
const buffer = tensorflow.getAllOpList();
console.log(buffer);
proto.load('./proto/op_def.proto', function(err, root){
    const OpList = root.lookupType('OpList');
    const oplist = OpList.decode(buffer);
    console.log(oplist.op[0]);
})