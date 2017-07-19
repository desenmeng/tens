const tf = require('tens')

//tf.__version__
console.log(tf.__version__)

const node1 = tf.constant(3.0, tf.float32)
const node2 = tf.constant(4.0) //also tf.float32 implicitly
const node3 = tf.add(node1, node2)
const sess = tf.Session()
console.log('node3', node3)
console.log('sess.run(node3)', sess.run(node3))
