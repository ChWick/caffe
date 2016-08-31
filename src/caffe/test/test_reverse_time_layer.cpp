#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/reverse_time_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class ReverseTimeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ReverseTimeLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 1)),
        blob_bottom_sequence_lengths_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    blob_bottom_vec_.push_back(blob_bottom_);
  }


  virtual ~ReverseTimeLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_sequence_lengths_;
    delete blob_top_;
  }

  void TestForwardAxis(
          int s0,
          int s1,
          int s2,
          int s3,
          const Dtype data_in[],
          const Dtype data_expected[],
          const Dtype sequenceLenghts[]) {
    LayerParameter layer_param;
    ReverseTimeParameter* reverse_time_param =
        layer_param.mutable_reverse_time_param();

    reverse_time_param->set_copy_remaining(true);

    shared_ptr<ReverseTimeLayer<Dtype> > layer(
          new ReverseTimeLayer<Dtype>(layer_param));

    // create dummy data and diff
    blob_bottom_->Reshape(s0, s1, s2, s3);
    blob_top_->ReshapeLike(*blob_bottom_);

    // create sequence lenghts
    const int num = blob_bottom_->shape(1) ;
    blob_bottom_vec_.push_back(blob_bottom_sequence_lengths_);
    std::vector<int> shape;
    shape.push_back(num);
    blob_bottom_sequence_lengths_->Reshape(shape);
    caffe_copy(shape[0], sequenceLenghts,
        blob_bottom_sequence_lengths_->mutable_cpu_data());

    // copy input data
    caffe_copy(blob_bottom_->count(), data_in,
               blob_bottom_->mutable_cpu_data());

    // setup layer
    layer->LayerSetUp(blob_bottom_vec_, blob_top_vec_);
    // Forward data
    layer->Forward(blob_bottom_vec_, blob_top_vec_);

    // Output of top must match the expected data
    EXPECT_EQ(blob_bottom_->count(), blob_top_->count());

    for (int i = 0; i < blob_top_->count(); ++i) {
      EXPECT_FLOAT_EQ(data_expected[i], blob_top_->cpu_data()[i]);
    }
  }

  void TestBackwardAxis(
          int s0,
          int s1,
          int s2,
          int s3,
          const Dtype diff_in[],
          const Dtype diff_expected[],
          const Dtype sequenceLenghts[]) {
    LayerParameter layer_param;
    ReverseTimeParameter* reverse_time_param =
        layer_param.mutable_reverse_time_param();
    reverse_time_param->set_copy_remaining(true);

    shared_ptr<ReverseTimeLayer<Dtype> > layer(
          new ReverseTimeLayer<Dtype>(layer_param));

    // create dummy data and diff
    blob_bottom_->Reshape(s0, s1, s2, s3);
    blob_top_->ReshapeLike(*blob_bottom_);

    // create sequence lenghts
    const int num = blob_bottom_->shape(1) ;
    blob_bottom_vec_.push_back(blob_bottom_sequence_lengths_);
    std::vector<int> shape;
    shape.push_back(num);
    blob_bottom_sequence_lengths_->Reshape(shape);
    caffe_copy(shape[0], sequenceLenghts,
        blob_bottom_sequence_lengths_->mutable_cpu_data());

    // copy input diff
    caffe_copy(blob_top_->count(), diff_in, blob_top_->mutable_cpu_diff());

    // setup layer
    layer->LayerSetUp(blob_bottom_vec_, blob_top_vec_);
    // Backward diff
    layer->Backward(blob_top_vec_, vector<bool>(1, true), blob_bottom_vec_);

    // Output of top must match the expected data
    EXPECT_EQ(blob_bottom_->count(), blob_top_->count());

    for (int i = 0; i < blob_top_->count(); ++i) {
      EXPECT_FLOAT_EQ(diff_expected[i], blob_bottom_->cpu_diff()[i]);
    }
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_sequence_lengths_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ReverseTimeLayerTest, TestDtypesAndDevices);

TYPED_TEST(ReverseTimeLayerTest, TestForwardAxis) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype data_in[5 * 2 * 1 * 3] = {
    1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30
  };

  // first axis must be inverted
  const Dtype data_expected[5 * 2 * 1 * 3] = {
    25, 26, 27, 16, 17, 18,
    19, 20, 21, 10, 11, 12,
    13, 14, 15, 4, 5, 6,
    7, 8, 9, 22, 23, 24,
    1, 2, 3, 28, 29, 30
  };

  // sequence lengths
  const Dtype sequence_lengths[2] = {
    5, 3
  };


  this->TestForwardAxis(5, 2, 1, 3, data_in, data_expected, sequence_lengths);
}

TYPED_TEST(ReverseTimeLayerTest, TestBackwardAxis) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype diff_in[5 * 2 * 1 * 3] = {
    100, 101, 102, 103, 104, 105,
    106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 122, 123,
    124, 125, 126, 127, 128, 129
  };

  // first axis must be inverted
  const Dtype diff_expected[5 * 2 * 1 * 3] = {
    124, 125, 126, 115, 116, 117, 
    118, 119, 120, 109, 110, 111,
    112, 113, 114, 103, 104, 105,
    106, 107, 108, 121, 122, 123,
    100, 101, 102, 127, 128, 129
  };

  // sequence lengths
  const Dtype sequence_lengths[2] = {
    5, 3
  };

  this->TestBackwardAxis(5, 2, 1, 3, diff_in, diff_expected, sequence_lengths);
}

}  // namespace caffe
