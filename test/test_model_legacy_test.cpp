// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include <fdeep/fdeep.hpp>

TEST_CASE("Keras 2 and Keras 3 graphs produce the same predictions")
{
    auto data = nlohmann::json::parse(R"({
        "image_data_format":"channels_last",
        "input_shapes":[[1],[1]], "output_shapes":[[1]],
        "trainable_params":{},
        "architecture":{"class_name":"Functional","config":{
            "name":"sum","layers":[
                {"class_name":"InputLayer","name":"a",
                 "config":{"name":"a","batch_input_shape":[null,1]},"inbound_nodes":[]},
                {"class_name":"InputLayer","name":"b",
                 "config":{"name":"b","batch_input_shape":[null,1]},"inbound_nodes":[]},
                {"class_name":"Add","name":"add","config":{"name":"add"},
                 "inbound_nodes":[[["a",0,0,{}],["b",0,0,{}]]]}
            ],
            "input_layers":[["a",0,0],["b",0,0]],"output_layers":[["add",0,0]]
        }}
    })");
    const fdeep::tensors inputs = {
        fdeep::tensor(fdeep::tensor_shape(1), fdeep::float_vec { 2 }),
        fdeep::tensor(fdeep::tensor_shape(1), fdeep::float_vec { 3 })
    };
    const auto predict = [&](const nlohmann::json& model_data) {
        const auto model = fdeep::read_model_from_string(model_data.dump(), false,
            fdeep::dev_null_logger);
        return model.predict(inputs).front().get_ignore_rank(fdeep::tensor_pos(0));
    };
    CHECK(predict(data) == 5);
    auto& layers = data["architecture"]["config"]["layers"];
    layers[0]["config"]["batch_shape"] = { nullptr, 1 };
    layers[1]["config"]["batch_shape"] = { nullptr, 1 };
    // A modern shape takes precedence if both spellings are present.
    layers[0]["config"]["batch_input_shape"] = { nullptr, 7 };
    layers[2]["inbound_nodes"] = nlohmann::json::parse(R"([{
        "args":[[
            {"config":{"keras_history":["a",0,0]}},
            {"config":{"keras_history":["b",0,0]}}
        ]],"kwargs":{}
    }])");
    CHECK(predict(data) == 5);
}

TEST_CASE("Keras 2 connections reject unsupported arguments")
{
    using fdeep::internal::create_node;
    CHECK_THROWS(create_node(nlohmann::json::parse(R"([["input",0,0]])")));
    CHECK_THROWS(create_node(nlohmann::json::parse(R"([["input",0,0,{"mask":true}]])")));
    CHECK_THROWS(create_node(nlohmann::json::parse(R"([["input",0,0,null]])")));
}
