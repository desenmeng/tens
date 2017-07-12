{
  "targets": [{
    "target_name": "tensorflow",
    "sources": [ "main.cc" ],
    "cflags!": [ "-fno-exceptions" ],
    "cflags_cc!": [ "-fno-exceptions" ],
    "xcode_settings": {
      "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
      "CLANG_CXX_LIBRARY": "libc++",
      "MACOSX_DEPLOYMENT_TARGET": "10.7"
    },
    "msvs_settings": {
      "VCCLCompilerTool": { "ExceptionHandling": 1 }
    },
    'include_dirs': [
      'tensorflow'
    ],
    'libraries': ['../lib/libtensorflow.so'],
  }]
}