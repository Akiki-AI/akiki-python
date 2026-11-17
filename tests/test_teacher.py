"""Teacher feedback must be usable for retraining without mutating source data."""

import json
import os
import subprocess
import sys
import threading
from collections import deque
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from akiki._specialist_training import _root, _text_hash, _write_json
from akiki.decision import Classification, DecisionModel, ModelPrecision, Specialist
from akiki.teacher import DecisionReview, Teacher, TeacherError, Verdict


def completion(value, *, finish_reason="stop"):
    content = value if isinstance(value, str) else json.dumps(value)
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


def assessment(verdict="incorrect", expected_label="task"):
    return {
        "verdict": verdict,
        "expected_label": expected_label,
        "reason": "Polite wording still requests an action.",
    }


@pytest.fixture
def server():
    state = SimpleNamespace(
        responses=deque(), requests=[], models=[{"id": "teacher-model"}], status=200, headers={}
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state.requests.append(("GET", self.path, dict(self.headers), None))
            self.reply({"data": state.models})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append(("POST", self.path, dict(self.headers), body))
            self.reply(state.responses.popleft() if state.responses else {})

        def reply(self, value):
            data = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(state.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for name, value in state.headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{httpd.server_port}"
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield state
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    specialist = Specialist(
        intents={"question", "task"},
        examples={"Where is the menu?": "question", "Save my file.": "task"},
        cache_dir=tmp_path / "cache",
    )
    model = DecisionModel()
    model.Specialists.add(specialist)
    state = SimpleNamespace(model=model, specialist=specialist, calls=[], predictions={})

    def classify(text, *, specialist):
        assert specialist is state.specialist
        state.calls.append(text)
        label = state.predictions.get(text, "question")
        return Classification(
            index=0 if label == "question" else 1,
            label=label,
            probabilities=(0.8, 0.2),
            logits=(1.4, 0.0),
        )

    monkeypatch.setattr(model, "classify", classify)
    yield state
    model.close()


def test_wrong_decision_generates_reviewable_supplemental_data(server, configured, tmp_path):
    server.responses.extend(
        [
            completion(assessment()),
            completion(
                {
                    "examples": [
                        {"text": "Please write my changes to disk.", "label": "task"},
                        {"text": "Could you preserve this document?", "label": "task"},
                    ]
                }
            ),
        ]
    )
    before = configured.specialist.examples.copy()
    teacher = Teacher(
        server.url + "/v1/",
        model="explicit-model",
        api_key="test-secret",
        task_description="Distinguish requests from questions.",
        intent_descriptions={"task": "Requests for an action."},
    )
    assert not server.requests
    result = teacher.review("Could you save my changes?", configured.model, examples_per_error=2)

    assert configured.calls == ["Could you save my changes?"]
    assert configured.specialist.examples == before
    (review,) = result.reviews
    assert review.verdict is Verdict.INCORRECT
    assert review.expected_label == "task"
    assert review.decision.label == "question"
    assert review.examples == result.examples
    assert set(result.examples.values()) == {"task"}
    assert len(result.examples) == 2
    assert [entry[1] for entry in server.requests] == ["/v1/chat/completions"] * 2
    assert server.requests[0][2]["Authorization"] == "Bearer test-secret"
    for _, _, _, payload in server.requests:
        assert payload["model"] == "explicit-model"
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["stream"] is False
        context = json.loads(payload["messages"][1]["content"])
        assert context["task"] == teacher.task_description
        assert set(context["labels"]) == configured.specialist.intents
        assert context["examples"]
    generation = json.loads(server.requests[1][3]["messages"][1]["content"])
    assert generation["expected_label"] == "task" and generation["count"] == 2

    output = tmp_path / "supplemental.jsonl"
    result.save_examples(output)
    assert {
        row["text"]: row["label"] for row in map(json.loads, output.read_text().splitlines())
    } == result.examples
    with pytest.raises(FileExistsError):
        result.save_examples(output)


def test_correct_and_uncertain_decisions_do_not_generate(server, configured):
    server.responses.extend(
        [
            completion(assessment("correct", "question")),
            completion(assessment("uncertain", None)),
        ]
    )
    teacher = Teacher(server.url)
    result = teacher.review(["Where is the menu?", "Maybe later?"], configured.model)

    assert [review.verdict for review in result.reviews] == [Verdict.CORRECT, Verdict.UNCERTAIN]
    assert result.examples == {}
    assert all(review.examples == {} for review in result.reviews)
    assert [request[0] for request in server.requests] == ["GET", "POST", "POST"]
    assert teacher.model == "teacher-model"
    assert all("Authorization" not in request[2] for request in server.requests)


def test_filters_training_holdouts_inputs_and_batch_duplicates(server, configured):
    validation = "Reserved validation text."
    test = "Reserved test text."
    configured.specialist.examples[validation] = "question"
    report = _root(configured.specialist) / "base" / "run" / "training.json"
    _write_json(
        report,
        {"split_text_sha256": {"validation": [_text_hash(validation)], "test": [_text_hash(test)]}},
    )
    _write_json(report.parent.parent / "latest.json", {"run": "run"})
    inputs = ["Could you save my changes?", "Could you close the editor?"]
    rows = [
        {"text": text, "label": "task"}
        for text in [
            "  SAVE   MY FILE. ",
            validation.lower(),
            test.upper(),
            "Extra reserved sample.",
            inputs[0],
            inputs[1],
            "Please save the current buffer.",
            "ＰＬＥＡＳＥ save the current buffer.",
        ]
    ]
    server.responses.extend(
        [
            completion(assessment()),
            completion({"examples": rows}),
            completion(assessment()),
            completion(
                {
                    "examples": [
                        {"text": "Please save the current buffer.", "label": "task"},
                        {"text": "Please exit this editor.", "label": "task"},
                    ]
                }
            ),
        ]
    )
    before = configured.specialist.examples.copy()
    result = Teacher(server.url, model="teacher").review(
        inputs, configured.model, excluded_examples={"Extra reserved sample.": "task"}
    )
    assert result.examples == {
        "Please save the current buffer.": "task",
        "Please exit this editor.": "task",
    }
    assert configured.specialist.examples == before
    for _, _, _, payload in server.requests:
        references = json.loads(payload["messages"][1]["content"])["examples"]
        assert validation not in {row["text"] for row in references}

    with pytest.raises(ValueError, match="reserved"):
        Teacher(server.url, model="teacher").review(validation.upper(), configured.model)
    assert len(server.requests) == 4


@pytest.mark.parametrize("models", [[], [{"id": "one"}, {"id": "two"}], [{"id": ""}]])
def test_model_discovery_requires_an_unambiguous_id(server, configured, models):
    server.models = models
    with pytest.raises(TeacherError, match="Pass model explicitly"):
        Teacher(server.url).review("Could you save?", configured.model)
    assert [request[0] for request in server.requests] == ["GET"]


@pytest.mark.parametrize(
    "response",
    [
        completion(assessment("correct", "task")),
        completion(assessment("incorrect", "question")),
        completion(assessment("uncertain", "task")),
        completion(assessment("incorrect", "new-label")),
        completion(assessment("maybe", None)),
        completion({"verdict": "incorrect", "expected_label": "task"}),
        completion("not JSON"),
        completion("[]"),
        completion("{}", finish_reason="length"),
        {"choices": []},
        {"choices": [None]},
        b"not HTTP JSON",
    ],
)
def test_invalid_assessments_never_generate_or_change_data(server, configured, response):
    server.responses.append(response)
    before = configured.specialist.examples.copy()
    with pytest.raises(TeacherError):
        Teacher(server.url, model="teacher").review("Could you save?", configured.model)
    assert len(server.requests) == 1
    assert configured.specialist.examples == before


@pytest.mark.parametrize(
    "generated",
    [
        {"examples": {"text": "Example", "label": "task"}},
        {"examples": [{"text": "A question?", "label": "question"}]},
        {"examples": [{"text": "", "label": "task"}]},
        {"examples": [{"text": 42, "label": "task"}]},
    ],
)
def test_invalid_generation_is_rejected(server, configured, generated):
    server.responses.extend([completion(assessment()), completion(generated)])
    before = configured.specialist.examples.copy()
    with pytest.raises(TeacherError):
        Teacher(server.url, model="teacher").review("Could you save?", configured.model)
    assert configured.specialist.examples == before


def test_conflicting_labels_across_errors_are_rejected(server, configured):
    configured.predictions["Where do I save?"] = "task"
    server.responses.extend(
        [
            completion(assessment()),
            completion({"examples": [{"text": "Same text", "label": "task"}]}),
            completion(assessment("incorrect", "question")),
            completion({"examples": [{"text": "same text", "label": "question"}]}),
        ]
    )
    with pytest.raises(TeacherError, match="conflicting labels"):
        Teacher(server.url, model="teacher").review(
            ["Could you save?", "Where do I save?"], configured.model
        )


def test_custom_api_base_and_plain_json_mode(server, configured):
    server.responses.append(
        completion("```json\n" + json.dumps(assessment("correct", "question")) + "\n```")
    )
    Teacher(server.url + "/custom/api", model="teacher", json_mode=False).review(
        "Where do I save?", configured.model
    )
    assert server.requests[0][1] == "/custom/api/chat/completions"
    assert "response_format" not in server.requests[0][3]


@pytest.mark.parametrize("status", [401, 503, 302])
def test_server_errors_and_redirects_fail_without_retry(server, configured, status):
    server.status = status
    server.headers = {"Location": server.url + "/elsewhere"}
    with pytest.raises(TeacherError, match=f"HTTP {status}"):
        Teacher(server.url, model="teacher").review("Could you save?", configured.model)
    assert len(server.requests) == 1


def test_timeout_is_reported(configured, monkeypatch):
    teacher = Teacher("http://localhost:8000", model="teacher", timeout=0.1)

    def time_out(request, *, timeout):
        assert timeout == 0.1
        raise TimeoutError("timed out")

    monkeypatch.setattr(teacher._opener, "open", time_out)
    with pytest.raises(TeacherError, match="Could not reach"):
        teacher.review("Could you save?", configured.model)


def test_selection_and_inputs_are_validated_before_requests(server, configured):
    teacher = Teacher(server.url)
    configured.model.Specialists.add(Specialist(intents={"other", "intent"}))
    with pytest.raises(ValueError, match="explicitly"):
        teacher.review("input", configured.model)
    with pytest.raises(ValueError, match="attached"):
        teacher.review("input", configured.model, specialist=Specialist())
    with pytest.raises(ValueError, match="nonempty strings"):
        teacher.review(["input", ""], configured.model, specialist=configured.specialist)
    with pytest.raises(ValueError, match="positive integer"):
        teacher.review(
            "input", configured.model, specialist=configured.specialist, examples_per_error=True
        )
    with pytest.raises(ValueError, match="reserved"):
        teacher.review(
            "reserved",
            configured.model,
            specialist=configured.specialist,
            excluded_examples=["reserved"],
        )
    assert teacher.review([], configured.model, specialist=configured.specialist).examples == {}
    assert not server.requests and not configured.calls


def test_verdict_is_a_plain_enum():
    decision = Classification(0, "question", (0.8, 0.2), (1.4, 0.0))
    review = DecisionReview("Where?", decision, Verdict.CORRECT, "question", "Clear question.", {})
    assert not isinstance(review.verdict, str)
    with pytest.raises(TypeError, match="Verdict member"):
        replace(review, verdict="correct")


def test_import_does_not_load_optional_runtimes():
    code = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'akiki_training', 'openai'}:
            raise AssertionError('Unexpected optional dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from akiki.teacher import Teacher
Teacher('http://localhost:8000', model='teacher')
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )


def test_feedback_can_continue_specialist_training(
    server, tmp_path, tiny_minilm, require_minilm_library
):
    pytest.importorskip("akiki_training.training")
    from akiki import training

    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    specialist = Specialist(
        intents={"go", "stop"},
        examples={
            "w4 w5": "go",
            "w4 w6": "go",
            "w4 w7": "go",
            "w4 w8": "go",
            "w20 w21": "stop",
            "w20 w22": "stop",
            "w20 w23": "stop",
            "w20 w24": "stop",
        },
        cache_dir=tmp_path / "specialist",
    )
    validation = {"w4 w9": "go", "w4 w10": "go", "w20 w25": "stop", "w20 w26": "stop"}
    test = {"w4 w11": "go", "w20 w27": "stop"}
    config = training.TrainingConfig(
        epochs=1, minimum_epochs=1, patience=1, batch_size=4, head_learning_rate=0.05
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        first = specialist.train(
            tiny_minilm.plain, config=config, validation_examples=validation, test_examples=test
        )
        original_report = (first.output_dir / "training.json").read_bytes()
        with DecisionModel(
            tiny_minilm.plain, model_precision=ModelPrecision.FP32, threads=1
        ) as model:
            model.Specialists.add(specialist)
            predicted = model.classify("w4 w12").label
            corrected = next(label for label in sorted(specialist.intents) if label != predicted)
            server.responses.extend(
                [
                    completion(assessment("incorrect", corrected)),
                    completion({"examples": [{"text": "w4 w13", "label": corrected}]}),
                ]
            )
            result = Teacher(server.url, model="teacher").review("w4 w12", model)
            assert result.reviews[0].decision.label == predicted
            output = tmp_path / "supplemental.jsonl"
            result.save_examples(output)
            assert {(row.text, row.label) for row in training.load_examples(output)} == set(
                result.examples.items()
            )

            specialist.examples.update(result.examples)
            second = specialist.train(
                model.base_model,
                config=config,
                continue_training=True,
                validation_examples=validation,
                test_examples=test,
            )
            assert second.output_dir != first.output_dir
            assert model.classify("w4 w13").label in specialist.intents
        assert (first.output_dir / "training.json").read_bytes() == original_report
        before = json.loads(original_report)["split_text_sha256"]
        after = json.loads((second.output_dir / "training.json").read_text())["split_text_sha256"]
        assert after["validation"] == before["validation"]
        assert after["test"] == before["test"]
        assert len(specialist.examples) == 9
    finally:
        torch.set_num_threads(previous_threads)
