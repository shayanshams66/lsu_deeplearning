#include <glog/logging.h>
#include "singa/neuralnet/neuron_layer.h"
#include "singa/utils/singleton.h"
#include "singa/utils/math_blob.h"
#include "singa/utils/singa_op.h"
#include <iostream>

using namespace std;

namespace singa {

using std::vector;

LstmLayer::~LstmLayer() {
  delete weight_f_hx_;
  delete weight_f_hh_;
  delete bias_f_;

  delete weight_i_hx_;
  delete weight_i_hh_;
  delete bias_i_;

  delete weight_o_hx_;
  delete weight_o_hh_;
  delete bias_o_;

  delete weight_c_hx_;
  delete weight_c_hh_;
  delete bias_c_;


  delete input_gate_;
  delete output_gate_;
  delete forget_gate_;
  delete new_memory_;
  delete memory_cell_;

}
int i=0;
void LstmLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  CHECK_LE(srclayers.size(), 2);
  const auto& src = srclayers[0]->data(this);
  batchsize_ = src.shape()[0];  // size of batch
  vdim_ = src.count() / (batchsize_);  // dimension of input

  hdim_ = layer_conf_.lstm_conf().dim_hidden();  // dimension of hidden state
  data_.Reshape(vector<int>{batchsize_, hdim_});
  if (mem_.count() == 0){
  //    printf("mem_.count()%d\n",mem_.count());
     mem_.Reshape(data_.shape());
//      printf("mem_.count2()%d\n",mem_.count());
    }
  //mem_.Reshape(vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_);
  //mem_.ReshapeLike(data_);
  gradvec_.push_back(new Blob<float>(grad_.shape()));

//  printf("%d, %d, %d\n", batchsize_, vdim_, hdim_);
i++;
printf("%d\n",i);

  // Initialize the parameters

  weight_i_hx_ = Param::Create(conf.param(0));
  weight_f_hx_ = Param::Create(conf.param(1));
  weight_c_hx_ = Param::Create(conf.param(2));
  weight_o_hx_ = Param::Create(conf.param(3));

  weight_i_hh_ = Param::Create(conf.param(4));
  weight_f_hh_ = Param::Create(conf.param(5));
  weight_c_hh_ = Param::Create(conf.param(6));
  weight_o_hh_ = Param::Create(conf.param(7));

  if (conf.param_size() > 8) {
    bias_i_ = Param::Create(conf.param(8));
    bias_f_ = Param::Create(conf.param(9));
    bias_c_ = Param::Create(conf.param(10));
    bias_o_ = Param::Create(conf.param(11));
  }

  weight_i_hx_->Setup(vector<int>{hdim_, vdim_});
  weight_f_hx_->Setup(vector<int>{hdim_, vdim_});
  weight_c_hx_->Setup(vector<int>{hdim_, vdim_});
  weight_o_hx_->Setup(vector<int>{hdim_, vdim_});

  weight_i_hh_->Setup(vector<int>{hdim_, hdim_});
  weight_f_hh_->Setup(vector<int>{hdim_, hdim_});
  weight_c_hh_->Setup(vector<int>{hdim_, hdim_});
  weight_o_hh_->Setup(vector<int>{hdim_, hdim_});

  if (conf.param_size() > 8) {
    bias_i_->Setup(vector<int>{hdim_});
    bias_f_->Setup(vector<int>{hdim_});
    bias_c_->Setup(vector<int>{hdim_});
    bias_o_->Setup(vector<int>{hdim_});
  }

  input_gate_ = new Blob<float>(batchsize_, hdim_);
  output_gate_ = new Blob<float>(batchsize_, hdim_);
  forget_gate_ = new Blob<float>(batchsize_, hdim_);
  new_memory_ = new Blob<float>(batchsize_, hdim_);
  memory_cell_ = new Blob<float>(batchsize_, hdim_);
//  printf("SETUP FINISHED\n");
}
//memory cell context
//virtual const Blob<float>& mem(const Layer* from) {
//    return mem_;
//  }
void LstmLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  CHECK_LE(srclayers.size(), 2);
//  printf("COMPUTE FEATURE\n");

  // Do transpose
  Blob<float> *w_f_hx_t = Transpose(weight_f_hx_->data());
  Blob<float> *w_f_hh_t = Transpose(weight_f_hh_->data());
  Blob<float> *w_i_hx_t = Transpose(weight_i_hx_->data());
  Blob<float> *w_i_hh_t = Transpose(weight_i_hh_->data());
  Blob<float> *w_o_hx_t = Transpose(weight_o_hx_->data());
  Blob<float> *w_o_hh_t = Transpose(weight_o_hh_->data());
  Blob<float> *w_c_hx_t = Transpose(weight_c_hx_->data());
  Blob<float> *w_c_hh_t = Transpose(weight_c_hh_->data());

  // Prepare the data input and the context
  const auto& src = srclayers[0]->data(this);
  const Blob<float> *context;
  const Blob<float> *memory_context;
  if (srclayers.size() == 1) {  // only have data input
    context = new Blob<float>(batchsize_, hdim_);
    memory_context = new Blob<float>(batchsize_,hdim_);
    //printf("firstcon\n");
    //printf("data1%d\n",data_.count() );
    //printf("data1%d\n",mem_.count() );
  } else {  // have data input & context & memory_context
    context = &srclayers[1]->data(this);
    //printf("con2\n");
    
      //if  (srclayers[1]->mem(this).shape()[0] == 0){
        //  memory_context= &srclayers[1]->data(this);
      //}else{
      //  memory_context= &srclayers[1]->mem(this);
    //  }

    memory_context= &srclayers[1]->mem(this);
    //printf("data2%d\n",data_.count());
    //printf("mem_2%d\n",mem_.count());
  }
  // Compute the input gate
  GEMM(1.0f, 0.0f, src, *w_i_hx_t, input_gate_);
  if (bias_i_ != nullptr)
    MVAddRow(1.0f, 1.0f, bias_i_->data(), input_gate_);
  GEMM(1.0f, 1.0f, *context, *w_i_hh_t, input_gate_);
  Map<op::Sigmoid<float>, float>(*input_gate_, input_gate_);
  
  // Compute the forget gate
  GEMM(1.0f, 0.0f, src, *w_f_hx_t, forget_gate_);
  if (bias_f_ != nullptr)
    MVAddRow(1.0f, 1.0f, bias_f_->data(), forget_gate_);
  GEMM(1.0f, 1.0f, *context, *w_f_hh_t, forget_gate_);
  Map<op::Sigmoid<float>, float>(*forget_gate_, forget_gate_);
  
  //Compute the output gate
  GEMM(1.0f, 0.0f, src, *w_o_hx_t, output_gate_);
  if (bias_o_ != nullptr)
    MVAddRow(1.0f, 1.0f, bias_o_->data(), output_gate_);
  GEMM(1.0f, 1.0f, *context, *w_o_hh_t, output_gate_);
  Map<op::Sigmoid<float>, float>(*output_gate_, output_gate_);  
  //compute memory
  GEMM(1.0f, 0.0f, src, *w_c_hx_t, new_memory_);
  if (bias_c_ != nullptr)
    MVAddRow(1.0f, 1.0f, bias_f_->data(), new_memory_);
  GEMM(1.0f, 1.0f, *context, *w_c_hh_t, new_memory_);
  
  //printf("count %d\n",*new_memory_->);


  //std::cout << "size of C " << op::Sigmoid<float>->count() << std::endl;
  Map<op::Sigmoid<float>, float>(*new_memory_, new_memory_);
  //compute memory context
  Mult(*memory_context, *forget_gate_, &mem_);
  Mult<float>(*output_gate_,*new_memory_, new_memory_);
  Add(mem_, *new_memory_, &mem_);
  //compute context
  Map<op::Tanh<float>, float>(mem_, memory_cell_);
  Mult(*memory_cell_, *output_gate_, &data_);

  // delete the pointers
  if (srclayers.size() == 1)
    delete context;
    //dete memory_context;
  delete w_i_hx_t;
  delete w_i_hh_t;
  delete w_f_hx_t;
  delete w_f_hh_t;
  delete w_o_hx_t;
  delete w_o_hh_t;
  delete w_c_hx_t;
  delete w_c_hh_t;
}
//int i=0;
void LstmLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  CHECK_LE(srclayers.size(), 2);
  AXPY(1.0f, *gradvec_[1], &grad_);
  AXPY(1.0f, *gradvec_[1], &mem_);
  float beta = 1.0f;  // agg param gradients
//printf("%d\n",i++);
  Layer* ilayer = srclayers[0];  // input layer
  Layer* clayer = nullptr;  // context layer
  // Prepare the data input and the context
  const Blob<float>& src = ilayer->data(this);
  const Blob<float> *context;
  const Blob<float> *memory_context;
  //printf("shayan");
  if (srclayers.size() == 1) {  // only have data input
    context = new Blob<float>(batchsize_, hdim_);
    memory_context = new Blob<float>(batchsize_,hdim_);
  } else {  // have data input & context
    clayer = srclayers[1];
    context = &(clayer->data(this));
     memory_context= &srclayers[1]->mem(this);
    //if  (&(clayer->mem(this)) == nullptr){
      //    memory_context= &srclayers[1]->data(this);
      //}else{
        //  memory_context= &srclayers[1]->mem(this);
      //}
  }
  Blob<float> digatedi(batchsize_, hdim_);
  Map<singa::op::SigmoidGrad<float>, float>(*input_gate_, &digatedi);

  Blob<float> dfgatedf(batchsize_, hdim_);
  Map<singa::op::SigmoidGrad<float>, float>(*forget_gate_, &dfgatedf);
  Blob<float> dogatedo(batchsize_, hdim_);
  Map<singa::op::SigmoidGrad<float>, float>(*output_gate_, &dogatedo);
  Blob<float> dnewmdc(batchsize_, hdim_);
  Map<singa::op::TanhGrad<float>, float>(*new_memory_, &dnewmdc);
  //lob<float> dLdcell(batchsize_, hdim_);
  Blob<float> dLdo(batchsize_, hdim_);
  Map<op::Tanh<float>, float>(*memory_context, &dLdo);
  Mult<float>(dLdo, grad_, &dLdo);
  Mult<float>(dLdo, dogatedo, &dLdo);


  Blob<float> dLdcell(batchsize_, hdim_);
  Map<singa::op::TanhGrad<float>, float>(*memory_context, &dLdcell);
  Mult<float>(dLdcell, grad_, &dLdcell);
  Mult<float>(dLdcell, *output_gate_, &dLdcell);
  //Mult<float>(dLdcell, dmcelldcell, &dLdcell);

  Blob<float> input_dLdcell(batchsize_, hdim_);
  Mult<float>(dLdcell, *new_memory_, &input_dLdcell);

  Blob<float> forget_dLdcell(batchsize_, hdim_);
  Mult<float>(dLdcell, *memory_context, &forget_dLdcell);

  Blob<float> newmemory_dLdcell(batchsize_, hdim_);
  Mult<float>(dLdcell, *input_gate_, &newmemory_dLdcell);

  Blob<float> mem_dLdcell(batchsize_, hdim_);
  Mult<float>(dLdcell, *memory_context, &mem_dLdcell);

  Blob<float> dLdi(batchsize_, hdim_);
  Mult<float>(input_dLdcell, digatedi, &dLdi);
  
  Blob<float> dLdf(batchsize_, hdim_);
  Mult<float>(forget_dLdcell, dfgatedf, &dLdf);

  Blob<float> dLdnewm(batchsize_, hdim_);
  Mult<float>(newmemory_dLdcell, dnewmdc, &dLdnewm);

  // Blob<float> dLdi(batchsize_, hdim_);
  //Mult<float>(dLdcell, *new_memory_, &dLdi);
  //Mult<float>(dLdi, digatedi, &dLdi);

  //Blob<float> dLdf(batchsize_, hdim_);
  //Mult<float>(dLdcell, *memory_context, &dLdf);
  //Mult<float>(dLdf, dfgatedf, &dLdi);

  //Blob<float> dLdc(batchsize_, hdim_);
  //Mult<float>(dLdcell, *input_gate_, &dLdc);
  //Mult<float>(dLdc, dnewmdc, &dLdc);


  // Compute gradients for parameters of inpute gate
  Blob<float> *dLdi_t = Transpose(dLdi);
  GEMM(1.0f, beta, *dLdi_t, src, weight_i_hx_->mutable_grad());
  GEMM(1.0f, beta, *dLdi_t, *context, weight_i_hh_->mutable_grad());
  if (bias_i_ != nullptr)
    MVSumRow<float>(1.0f, beta, dLdi, bias_i_->mutable_grad());
  delete dLdi_t;


  // Compute gradients for parameters of forget gate
  Blob<float> *dLdf_t = Transpose(dLdf);
  GEMM(1.0f, beta, *dLdf_t, src, weight_f_hx_->mutable_grad());
  GEMM(1.0f, beta, *dLdf_t, *context, weight_f_hh_->mutable_grad());
  if (bias_f_ != nullptr)
    MVSumRow(1.0f, beta, dLdf, bias_f_->mutable_grad());
  delete dLdf_t;


  // Compute gradients for parameters of outputgate
  Blob<float> *dLdo_t = Transpose(dLdo);
  GEMM(1.0f, beta, *dLdo_t, src, weight_o_hx_->mutable_grad());
  GEMM(1.0f, beta, *dLdo_t, *context, weight_o_hh_->mutable_grad());
  if (bias_o_ != nullptr)
    MVSumRow(1.0f, beta, dLdo, bias_o_->mutable_grad());
  delete dLdo_t;



 // Compute gradients for parameters of new memory
  Blob<float> *dLdc_t = Transpose(dLdnewm);
  //printf("size%d\n",dLdc_t->count());
  //printf("ternery%d\n",dLdc_t->transpose());
  //printf("ternary%d\n",dLdc_t->shape(1));

  GEMM(1.0f, beta, *dLdc_t, src, weight_c_hx_->mutable_grad());

  //printf("c_weight_shape %d\n", weight_c_hh_->mutable_grad()->shape(0));
  //printf("c_weight %d\n",weight_c_hh_->mutable_grad()->count());
  
  GEMM(1.0f, beta, *dLdc_t, *context, weight_c_hh_->mutable_grad());
  //printf("GEMM\n");
  if (bias_c_ != nullptr)
    MVSumRow(1.0f, beta, dLdnewm, bias_c_->mutable_grad());
  delete dLdc_t;

   

   // Compute gradients for data input layer
  if (srclayers[0]->mutable_grad(this) != nullptr) {
    GEMM(1.0f, 1.0f, dLdnewm, weight_c_hx_->data(), ilayer->mutable_grad(this));
    GEMM(1.0f, 1.0f, dLdo, weight_o_hx_->data(), ilayer->mutable_grad(this));
    GEMM(1.0f, 1.0f, dLdf, weight_f_hx_->data(), ilayer->mutable_grad(this));
    GEMM(1.0f, 1.0f, dLdi, weight_i_hx_->data(), ilayer->mutable_grad(this));
  }

  if (clayer != nullptr && clayer->mutable_grad(this) != nullptr) {
    // Compute gradients for context layer
    GEMM(1.0f, 1.0f, dLdnewm, weight_c_hh_->data(),clayer->mutable_grad(this));
    GEMM(1.0f, 1.0f, dLdi, weight_i_hh_->data(), clayer->mutable_grad(this));
    GEMM(1.0f, 1.0f, dLdf, weight_f_hh_->data(), clayer->mutable_grad(this));
    GEMM(1.0f, 1.0f, dLdo, weight_o_hh_->data(), clayer->mutable_grad(this));
    Add(clayer->grad(this), *input_gate_, clayer->mutable_grad(this));
  }
  ;
  if (srclayers.size() == 1)
    delete context;
    //delete memory_context;
}

}  // namespace singa












