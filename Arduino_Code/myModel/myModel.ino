
// Import TensorFlowLite libraries

#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

//My model
#include "myModel.h"

//Figure out whats going on in the model
#define DEBUG 1

//Put other setting here if needed


//TFLite globals for Arduino compatiability
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

constexpr int kTensorArenaSize = 1 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

} //namespace ends

void setup() {
 
#if DEBUG
  while(!Serial);
#endif

//Set up logging report to Serial
static tflite::MicroErrorReporter micro_error_reporter;
error_reporter = &micro_error_reporter;

// Map the model into usuable data structure
model = tflite::GetModel(myModel_tflite);
if (model->version()  != TFLITE_SCHEMA_VERSION) {
  error_reporter->Report("Model version does not match schema");
  while(1);
}

//Pull in all operations
static tflite::AllOpsResolver resolver;

//Build an interpreter to run model
static tflite::MicroInterpreter static_interpreter(
  model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;


//Allocate memory from tensor_arena for model's tensors
TfLiteStatus allocate_status = interpreter->AllocateTensors();
if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    while(1);
}

//Assign model inout and output buffers (tensors) to pinters
model_input = interpreter->input(0);
model_output = interpreter->output(0);

//Get information about the memory area to use for model input
#if DEBUG
  Serial.print("Number of dimensions: ");
  Serial.println(model_input->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(model_input->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(model_input->dims->data[1]);
  Serial.print("Input type: ");
  Serial.println(model_input->type);
#endif

}

  
void loop() {
int8_t x_val = 100;
model_input->data.int8[0] = x_val;

TfLiteStatus invoke_status = interpreter->Invoke();

  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on input: %f\n", x_val);
  }

int8_t y_val = model_output->data.int8[0];

 Serial.println(y_val);
 delay(5000);

}
