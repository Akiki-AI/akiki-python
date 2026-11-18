"""Run VAD inference with the real CPU engines and local test data."""

import hashlib
import os

import numpy as np
import pytest

from akiki.audio import _ten_vad_model, vad

pytestmark = pytest.mark.skipif(
    os.environ.get("AKIKI_TEST_VAD_RUNTIMES") != "1",
    reason="Run the native and ONNX Runtime matrix with AKIKI_TEST_VAD_RUNTIMES=1",
)


@pytest.fixture
def ten_vad_onnx(tmp_path, monkeypatch):
    """Generate a TEN-shaped ONNX graph with synthetic weights."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(27)
    nodes, initializers = [], []

    def initializer(name, values):
        initializers.append(numpy_helper.from_array(np.asarray(values), name=name))
        return name

    def weight(name, shape):
        return initializer(name, rng.uniform(-0.15, 0.15, shape).astype(np.float32))

    def node(kind, inputs, output, **attributes):
        nodes.append(helper.make_node(kind, inputs, [output], **attributes))
        return output

    prefix = "StatefulPartitionedCall/vad_model/"
    axis_zero = initializer("axis_zero", np.array([0], np.int64))
    axis_one = initializer("axis_one", np.array([1], np.int64))
    features = node("Unsqueeze", ["features", axis_one], "features_nchw")
    for layer, index, depth_shape, point, point_shape, group, pads in (
        (
            "separable_conv2d",
            178,
            (1, 1, 3, 3),
            "separable_conv2d/ReadVariableOp_1:0",
            (16, 1, 1, 1),
            1,
            [0, 0, 0, 0],
        ),
        (
            "separable_conv1d",
            179,
            (16, 1, 1, 3),
            "ExpandDims_2:0",
            (16, 16, 1, 1),
            16,
            [0, 1, 0, 1],
        ),
        (
            "separable_conv1d_1",
            180,
            (16, 1, 1, 3),
            "ExpandDims_2:0",
            (16, 16, 1, 1),
            16,
            [0, 0, 0, 1],
        ),
    ):
        depth = weight(f"const_fold_opt__{index}", depth_shape)
        depth_out = node(
            "Conv",
            [features, depth],
            f"depth_{index}",
            group=group,
            pads=pads,
            strides=[1, 1 if index == 178 else 2],
        )
        projection = weight(f"{prefix}{layer}/{point}", point_shape)
        bias = weight(f"{prefix}{layer}/BiasAdd/ReadVariableOp:0", (16,))
        conv = node("Conv", [depth_out, projection, bias], f"point_{index}")
        features = node("Relu", [conv], f"relu_{index}")
        if index == 178:
            features = node("MaxPool", [features], "pooled", kernel_shape=[1, 3], strides=[1, 2])

    ordered = node("Transpose", [features], "frequency_major", perm=[0, 2, 3, 1])
    lstm_shape = initializer("lstm_shape", np.array([1, 1, 80], np.int64))
    features = node("Reshape", [ordered, lstm_shape], "lstm_input")
    state_names = ("hidden1", "cell1", "hidden2", "cell2")
    hidden_outputs = []
    for index, (weight_name, recurrent_name, bias_name, inputs) in enumerate(
        (("W0__70", "R0__71", "B0__72", 80), ("W0__99", "R0__100", "B0__101", 64))
    ):
        hidden_name, cell_name = state_names[2 * index : 2 * index + 2]
        hidden = node("Unsqueeze", [hidden_name, axis_zero], f"{hidden_name}_3d")
        cell = node("Unsqueeze", [cell_name, axis_zero], f"{cell_name}_3d")
        hidden_out, cell_out = f"{hidden_name}_out_3d", f"{cell_name}_out_3d"
        nodes.append(
            helper.make_node(
                "LSTM",
                [
                    features,
                    weight(weight_name, (1, 256, inputs)),
                    weight(recurrent_name, (1, 256, 64)),
                    weight(bias_name, (1, 512)),
                    "",
                    hidden,
                    cell,
                ],
                [f"lstm_sequence{index}", hidden_out, cell_out],
                hidden_size=64,
            )
        )
        node("Squeeze", [hidden_out, axis_zero], f"{hidden_name}_out")
        node("Squeeze", [cell_out, axis_zero], f"{cell_name}_out")
        hidden_outputs.append(hidden_out)
        features = hidden_out

    features = node("Concat", hidden_outputs[::-1], "joined", axis=2)
    for layer, inputs, outputs in (("dense_3", 128, 32), ("dense_5", 32, 1)):
        projection = weight(f"{prefix}{layer}/Tensordot/ReadVariableOp:0", (inputs, outputs))
        bias = weight(f"{prefix}{layer}/BiasAdd/ReadVariableOp:0", (outputs,))
        features = node("MatMul", [features, projection], f"{layer}_projection")
        features = node("Add", [features, bias], f"{layer}_output")
        if layer == "dense_3":
            features = node("Relu", [features], "dense_relu")
    node("Sigmoid", [features], "probability")

    def value(name, shape):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    graph = helper.make_graph(
        nodes,
        "synthetic-ten-vad",
        [value("features", [1, 3, 41]), *(value(name, [1, 64]) for name in state_names)],
        [value("probability", [1, 1, 1]), *(value(f"{name}_out", [1, 64]) for name in state_names)],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    path = tmp_path / "ten-vad.onnx"
    onnx.save(model, path)

    # The production converter pins the original graph. Permit only this test graph here.
    for tensor in graph.initializer:
        if tensor.data_type == TensorProto.FLOAT:
            tensor.ClearField("raw_data")
    monkeypatch.setattr(
        _ten_vad_model, "_GRAPH_SHA256", hashlib.sha256(graph.SerializeToString()).hexdigest()
    )

    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(768) / 768)
    arrays = (
        ("AUP_AED_FEATURE_MEANS", np.zeros(41)),
        ("AUP_AED_FEATURE_STDS", np.ones(41)),
        ("AUP_AED_STFTWindow_Hann768", window),
    )
    (tmp_path / "coeff.h").write_text(
        "".join(
            f"float {name}[{len(values)}] = {{{','.join(map(str, values))}}};\n"
            for name, values in arrays
        )
    )
    return path


@pytest.mark.parametrize("model", list(vad.Model))
def test_vad_scores_and_resets_on_both_cpu_runtimes(model, request, tmp_path):
    is_ten = model is vad.Model.TEN_VAD
    model_path = request.getfixturevalue("ten_vad_onnx") if is_ten else None
    options = dict(
        model=model,
        model_path=model_path,
        model_precision=vad.ModelPrecision.FP32,
        cache_dir=tmp_path / "cache",
    )
    boundary = (1875 if is_ten else 9) * 256
    audio = np.random.default_rng(28).uniform(-0.3, 0.3, boundary + 256 + 17).astype(np.float32)
    with (
        vad.VAD(engine=vad.Engine.AKIKI, **options) as native,
        vad.VAD(engine=vad.Engine.ONNX, **options) as onnx,
    ):
        assert native.engine is vad.Engine.AKIKI
        assert onnx.engine is vad.Engine.ONNX
        assert onnx.session.get_providers() == ["CPUExecutionProvider"]
        results = []
        for detector in (native, onnx):
            assert detector.push(audio[:37]) == []
            frames = detector.push(audio[37 : boundary - 256])
            assert np.any(detector.state)
            frames += detector.push(audio[boundary - 256 : boundary])
            if is_ten:
                np.testing.assert_array_equal(detector.state, 0)
            frames += detector.push(audio[boundary:])
            assert len(detector.pending) == 17
            assert np.any(detector.state)
            assert len(frames) == boundary // 256 + 1
            assert frames[0].start_seconds == 0
            assert frames[-1].end_seconds == (boundary + 256) / 16000
            detector.reset()
            assert detector.push(audio[:256]) == frames[:1]
            results.append(frames)

    native_frames, onnx_frames = results
    np.testing.assert_allclose(
        [frame.probability for frame in native_frames],
        [frame.probability for frame in onnx_frames],
        atol=2e-5,
        rtol=1e-5,
    )
    assert [frame.speech for frame in native_frames] == [frame.speech for frame in onnx_frames]
