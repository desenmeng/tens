#include <node_api.h>
#include <tensorflow/c/c_api.h>
#include <tensorflow/core/platform/logging.h>

napi_value Version(napi_env env, napi_callback_info info)
{
    napi_status status;

    // size_t argc = 1;
    napi_value result;
    // status = napi_get_cb_info(env, info, &argc, argv, 0, 0);
    // if(status != napi_ok || argc < 1)
    // {
    //     napi_throw_type_error(env, "Wrong number of arguments");
    //     status = napi_get_undefined(env, argv);
    // }
    const char * version = TF_Version();
    size_t len  = strlen(version);
    // size_t versionL = version.length;
    // status = napi_get_value_string_utf8(env, argv[0], version, len, nullptr);
    status = napi_create_string_utf8(env, version, len, &result);
    if(status != napi_ok)
    {
        napi_throw_type_error(env, "get version error");
    }
    return result;
}

void Init(napi_env env, napi_value exports, napi_value module, void* priv)
{
    napi_status status;
    napi_property_descriptor desc =
        { "version", 0, Version, 0, 0, 0, napi_default, 0 };
    status = napi_define_properties(env, exports, 1, &desc);
}

NAPI_MODULE(addon, Init)