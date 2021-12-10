#include<torch/extension.h>

#include<iostream>
#include<vector>

namespace F = torch::nn::functional;

std::vector<torch::Tensor> myconv2d_forward(
	torch::Tensor input,
	torch::Tensor weight,
	torch::Tensor bias
) {
	auto output = F::conv2d(input, weight,
		F::Conv2dFuncOptions().bias(bias)
		);
	return { output };
}

std::vector<torch::Tensor> myconv2d_backward(
	torch::Tensor grad_output,
	torch::Tensor input,
	torch::Tensor weight,
	torch::Tensor bias
) {
	auto kernal_height = weight.size(2);
	auto kernal_width = weight.size(3);
	auto grad_input = F::conv2d(
		grad_output,
		weight.rot90(2, { (2),(3) }).transpose(0, 1),
		F::Conv2dFuncOptions().padding({ kernal_width - 1, kernal_height - 1 })
	);
	auto grad_weight = F::conv2d(
		input.transpose(0, 1),
		grad_output.transpose(0, 1)
	).transpose(0, 1);
	auto grad_bias = grad_output.sum({ 0,2,3 });
	return { grad_input, grad_weight, grad_bias };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &myconv2d_forward, "myconv2d forward");
	m.def("backward", &myconv2d_backward, "myconv2d backward");
}