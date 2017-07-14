#include <node_api.h>
#include <tensorflow/c/c_api.h>
#include <tensorflow/core/platform/logging.h>
#include <vector>

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

typedef std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)>
    unique_tensor_ptr;

static void Int32Deallocator(void* data, size_t, void* arg) {
  delete[] static_cast<tensorflow::int32*>(data);
}

TF_Operation* Const(TF_Tensor* t, TF_Graph* graph, TF_Status* s,
                    const char* name = "const") {
  TF_OperationDescription* desc = TF_NewOperation(graph, "Const", name);
  TF_SetAttrTensor(desc, "value", t, s);
  if (TF_GetCode(s) != TF_OK) return nullptr;
  TF_SetAttrType(desc, "dtype", TF_TensorType(t));
  return TF_FinishOperation(desc, s);
}

static TF_Tensor* Int32Tensor(tensorflow::int32 v) {
  const int num_bytes = sizeof(tensorflow::int32);
  tensorflow::int32* values = new tensorflow::int32[1];
  values[0] = v;
  return TF_NewTensor(TF_INT32, nullptr, 0, values, num_bytes,
                      &Int32Deallocator, nullptr);
}

TF_Operation* Add(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                  TF_Status* s, const char* name = "add") {
  TF_OperationDescription* desc = TF_NewOperation(graph, "AddN", name);
  TF_Output add_inputs[2] = {{l, 0}, {r, 0}};
  TF_AddInputList(desc, add_inputs, 2);
  return TF_FinishOperation(desc, s);
}

TF_Operation* ScalarConst(tensorflow::int32 v, TF_Graph* graph, TF_Status* s,
                          const char* name = "scalar") {
  unique_tensor_ptr tensor(Int32Tensor(v), TF_DeleteTensor);
  return Const(tensor.get(), graph, s, name);
}

TF_Operation* Placeholder(TF_Graph* graph, TF_Status* s,
                          const char* name = "feed") {
  TF_OperationDescription* desc = TF_NewOperation(graph, "Placeholder", name);
  TF_SetAttrType(desc, "dtype", TF_INT32);
  return TF_FinishOperation(desc, s);
}

class CSession {
 public:
  CSession(TF_Graph* graph, TF_Status* s) {
    TF_SessionOptions* opts = TF_NewSessionOptions();
    session_ = TF_NewSession(graph, opts, s);
    TF_DeleteSessionOptions(opts);
  }

  explicit CSession(TF_Session* session) : session_(session) {}

  ~CSession() {
    TF_Status* s = TF_NewStatus();
    CloseAndDelete(s);
    // EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_DeleteStatus(s);
  }

  void SetInputs(std::vector<std::pair<TF_Operation*, TF_Tensor*>> inputs) {
    DeleteInputValues();
    inputs_.clear();
    for (const auto& p : inputs) {
      inputs_.emplace_back(TF_Output{p.first, 0});
      input_values_.emplace_back(p.second);
    }
  }

  void SetOutputs(std::initializer_list<TF_Operation*> outputs) {
    ResetOutputValues();
    outputs_.clear();
    for (TF_Operation* o : outputs) {
      outputs_.emplace_back(TF_Output{o, 0});
    }
    output_values_.resize(outputs_.size());
  }

  void SetOutputs(const std::vector<TF_Output>& outputs) {
    ResetOutputValues();
    outputs_ = outputs;
    output_values_.resize(outputs_.size());
  }

  void SetTargets(std::initializer_list<TF_Operation*> targets) {
    targets_.clear();
    for (TF_Operation* t : targets) {
      targets_.emplace_back(t);
    }
  }

  void Run(TF_Status* s) {
    if (inputs_.size() != input_values_.size()) {
      // ADD_FAILURE() << "Call SetInputs() before Run()";
      return;
    }
    ResetOutputValues();
    output_values_.resize(outputs_.size(), nullptr);

    const TF_Output* inputs_ptr = inputs_.empty() ? nullptr : &inputs_[0];
    TF_Tensor* const* input_values_ptr =
        input_values_.empty() ? nullptr : &input_values_[0];

    const TF_Output* outputs_ptr = outputs_.empty() ? nullptr : &outputs_[0];
    TF_Tensor** output_values_ptr =
        output_values_.empty() ? nullptr : &output_values_[0];

    TF_Operation* const* targets_ptr =
        targets_.empty() ? nullptr : &targets_[0];

    TF_SessionRun(session_, nullptr, inputs_ptr, input_values_ptr,
                  inputs_.size(), outputs_ptr, output_values_ptr,
                  outputs_.size(), targets_ptr, targets_.size(), nullptr, s);

    DeleteInputValues();
  }

  void CloseAndDelete(TF_Status* s) {
    DeleteInputValues();
    ResetOutputValues();
    if (session_ != nullptr) {
      TF_CloseSession(session_, s);
      // EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
      TF_DeleteSession(session_, s);
      session_ = nullptr;
    }
  }

  TF_Tensor* output_tensor(int i) { return output_values_[i]; }

 private:
  void DeleteInputValues() {
    for (int i = 0; i < input_values_.size(); ++i) {
      TF_DeleteTensor(input_values_[i]);
    }
    input_values_.clear();
  }

  void ResetOutputValues() {
    for (int i = 0; i < output_values_.size(); ++i) {
      if (output_values_[i] != nullptr) TF_DeleteTensor(output_values_[i]);
    }
    output_values_.clear();
  }

  TF_Session* session_;
  std::vector<TF_Output> inputs_;
  std::vector<TF_Tensor*> input_values_;
  std::vector<TF_Output> outputs_;
  std::vector<TF_Tensor*> output_values_;
  std::vector<TF_Operation*> targets_;
};

napi_value SessionRun(napi_env env, napi_callback_info info)
{
    napi_status status;
    napi_value result;
    TF_Status* s = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    TF_Operation* feed = Placeholder(graph, s);
    TF_Operation* two = ScalarConst(2, graph, s);
    TF_Operation* add = Add(feed, two, graph, s);
    // Create a session for this graph.
    CSession csession(graph, s);
    // Run the graph.
    csession.SetInputs({{feed, Int32Tensor(3)}});
    csession.SetOutputs({add});
    csession.Run(s);
    TF_Tensor* out = csession.output_tensor(0);
    int32_t* input = static_cast<tensorflow::int32*>(TF_TensorData(out));
    // status = napi_get_value_int32(env, result, output_contents);
    status = napi_create_number(env, *input, &result);
    if(status != napi_ok)
    {
      napi_throw_type_error(env, "run error");
    }
    return result;
}

void Init(napi_env env, napi_value exports, napi_value module, void* priv)
{
    napi_status status;
    napi_property_descriptor desc =
        { "version", 0, Version, 0, 0, 0, napi_default, 0 };
    status = napi_define_properties(env, exports, 1, &desc);
    napi_property_descriptor desc1 =
        { "sessionRun", 0, SessionRun, 0, 0, 0, napi_default, 0 };
    status = napi_define_properties(env, exports, 1, &desc1);
    
}

NAPI_MODULE(tensorflow, Init)